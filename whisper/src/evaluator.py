from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import evaluate
import json
import re
import numpy as np
from sklearn.metrics import f1_score
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


def parse_multitask_output(
    text: str,
    special_tokens: Sequence[str],  # kept for backward compatibility; unused
    bos_token: str | None,
    eos_token: str | None,
) -> Tuple[str | None, str | None]:
    """
    Parse a JSON-formatted multitask output into (intent, transcript_text).

    Expected structure (whitespace and ordering may vary):

      <|startoftranscript|>{
        "Text": "...",
        "Domain": "telephony",
        "Intent": "scam"
      }<|endoftext|>
    """
    # Strip BOS/EOS wrappers if present
    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
    if eos_token and eos_token in text:
        text = text.split(eos_token, 1)[0]

    # Extract JSON substring
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, None

    json_part = text[start : end + 1]
    try:
        obj = json.loads(json_part)
    except Exception:
        # Fallback: try to recover intent/text via regex even if JSON is malformed
        intent_match = re.search(r'"Intent"\s*:\s*"([^"]+)"', text)
        text_match = re.search(r'"Text"\s*:\s*"([^"]+)"', text)
        intent = intent_match.group(1).strip() if intent_match else None
        transcript = text_match.group(1).strip() if text_match else None
        if not intent and not transcript:
            return None, None
        return (intent or None), (transcript or None)

    intent = obj.get("Intent")
    transcript = obj.get("Text")
    if isinstance(intent, str):
        intent = intent.strip() or None
    else:
        intent = None
    if isinstance(transcript, str):
        transcript = transcript.strip() or None
    else:
        transcript = None

    return intent, transcript


def _canonical_intent_label(intent: str | None) -> str | None:
    """
    Map arbitrary intent text to a canonical label string ("scam" or "non_scam"),
    using simple heuristics. This makes evaluation robust to minor formatting
    differences in the generated JSON.
    """
    if intent is None:
        return None
    t = intent.strip().lower()
    if not t:
        return None

    # Non-scam heuristics first to avoid matching the substring "scam" inside.
    if "non_scam" in t or "non scam" in t or "not_scam" in t or "not scam" in t:
        return "non_scam"
    if "scam" in t or "fraud" in t:
        return "scam"
    return None


def build_compute_metrics_fn(
    processor: WhisperProcessor,
    special_tokens: Sequence[str],
    label2token: Mapping[str, str],
) -> Any:
    """
    Create a compute_metrics function for Seq2SeqTrainer that:
    - Extracts intents and transcripts from JSON-formatted sequences.
    - Computes F1 over intents.
    - Computes WER over transcripts.
    """
    wer_metric = evaluate.load("wer")

    bos_token = processor.tokenizer.bos_token or "<|startoftranscript|>"
    eos_token = processor.tokenizer.eos_token or "<|endoftext|>"

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        predictions, label_ids = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_str = processor.batch_decode(
            predictions,
            skip_special_tokens=False,
        )

        # Replace -100 with pad token id so that batch_decode works
        label_ids_clean = np.where(
            label_ids != -100,
            label_ids,
            processor.tokenizer.pad_token_id,
        )
        label_str = processor.batch_decode(
            label_ids_clean,
            skip_special_tokens=False,
        )

        pred_intents: List[str] = []
        true_intents: List[str] = []
        pred_transcripts: List[str] = []
        true_transcripts: List[str] = []

        # Log a few sample predictions for debugging
        num_debug = min(5, len(pred_str))
        for i in range(num_debug):
            logger.info(
                "EVAL SAMPLE %d | pred: %.200s | label: %.200s",
                i, pred_str[i], label_str[i],
            )

        for p, t in zip(pred_str, label_str):
            p_intent, p_text = parse_multitask_output(
                p,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            t_intent, t_text = parse_multitask_output(
                t,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )

            # Canonicalize to {scam, non_scam} labels
            true_lbl = _canonical_intent_label(t_intent)
            pred_lbl = _canonical_intent_label(p_intent)

            if true_lbl is not None:
                true_intents.append(true_lbl)
                # Treat unknown/invalid predictions as a special wrong label
                pred_intents.append(pred_lbl if pred_lbl is not None else "__invalid__")

            if t_text is not None:
                true_transcripts.append(t_text)
                pred_transcripts.append(p_text or "")

        # Intent F1 (single-label classification)
        if true_intents:
            intent_f1 = f1_score(
                true_intents,
                pred_intents,
                average="macro",
                zero_division=0.0,
            )
        else:
            intent_f1 = 0.0

        # WER on transcripts – filter out empty references
        filtered_pairs = [
            (p, t)
            for p, t in zip(pred_transcripts, true_transcripts)
            if t is not None and t.strip() != ""
        ]

        if filtered_pairs:
            f_pred, f_true = zip(*filtered_pairs)
            wer = float(
                wer_metric.compute(
                    predictions=list(f_pred),
                    references=list(f_true),
                )
            )
        else:
            wer = 0.0

        logger.info(
            "EVAL SUMMARY | n_samples=%d | n_valid_intents=%d | n_invalid_preds=%d | f1=%.4f | wer=%.4f",
            len(pred_str),
            len(true_intents),
            sum(1 for p in pred_intents if p == "__invalid__"),
            float(intent_f1),
            wer,
        )

        return {
            "intent_f1_macro": float(intent_f1),
            "wer": wer,
        }

    return compute_metrics

