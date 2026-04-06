from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import evaluate
import numpy as np
from sklearn.metrics import f1_score
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


def _strip_whisper_prompt_tokens(text: str) -> str:
    # Robust against older checkpoints / generation configs that still leak
    # Whisper prompt tokens like <|translate|> or <|notimestamps|>.
    return re.sub(r"<\|[^|]+?\|>", "", text)


def _json_get_case_insensitive(obj: Mapping[str, Any], key: str) -> Any:
    for current_key, value in obj.items():
        if isinstance(current_key, str) and current_key.lower() == key.lower():
            return value
    return None


def _canonical_intent_label(intent: str | None) -> str | None:
    """
    Map arbitrary intent text to a canonical label string.
    """
    if intent is None:
        return None

    text = intent.strip().lower()
    if not text:
        return None

    # Check the negative class first to avoid matching the substring "scam"
    # inside labels like "non_scam".
    if "non_scam" in text or "non scam" in text or "not_scam" in text or "not scam" in text:
        return "non_scam"
    if text == "ham":
        return "non_scam"
    if "scam" in text or "fraud" in text:
        return "scam"
    return None


def parse_multitask_output(
    text: str,
    special_tokens: Sequence[str],  # kept for backward compatibility; unused
    bos_token: str | None,
    eos_token: str | None,
) -> Tuple[str | None, str | None]:
    """
    Parse a JSON-formatted multitask output into (intent, transcript_text).

    Expected structure:
      <|startoftranscript|>{"Text":"...","Intent":"scam"}<|endoftext|>

    The parser is intentionally tolerant because malformed JSON is itself a
    useful debugging signal for this project.
    """
    del special_tokens

    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
    if eos_token and eos_token in text:
        text = text.split(eos_token, 1)[0]
    text = _strip_whisper_prompt_tokens(text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_part = text[start : end + 1]
        try:
            obj = json.loads(json_part)
            intent = _json_get_case_insensitive(obj, "Intent")
            transcript = _json_get_case_insensitive(obj, "Text")

            if isinstance(intent, str):
                intent = intent.strip() or None
            else:
                intent = None

            if isinstance(transcript, str):
                transcript = transcript.strip() or None
            else:
                transcript = None

            return intent, transcript
        except Exception:
            pass

    intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    text_match = re.search(r'"text"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    intent = intent_match.group(1).strip() if intent_match else None
    transcript = text_match.group(1).strip() if text_match else None
    if intent or transcript:
        return intent or None, transcript or None

    # Recovery path for heavily malformed outputs such as:
    #   "scam", "telephony", "telephony", ...
    quoted_values = re.findall(r'"([^"]+)"', text)
    for value in quoted_values:
        if _canonical_intent_label(value) is not None:
            cleaned = value.strip()
            return cleaned or None, None

    return None, None


def build_compute_metrics_fn(
    processor: WhisperProcessor,
    special_tokens: Sequence[str],
    label2token: Mapping[str, str],
) -> Any:
    """
    Create a compute_metrics function for Seq2SeqTrainer that:
    - Extracts intents and transcripts from JSON-formatted sequences.
    - Computes macro F1 over intents.
    - Computes WER over transcripts.
    - Reports how often a valid intent could be recovered from predictions.
    """
    del label2token

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
        valid_pred_intent_count = 0

        num_debug = min(5, len(pred_str))
        for i in range(num_debug):
            logger.info(
                "EVAL SAMPLE %d | pred: %.200s | label: %.200s",
                i,
                pred_str[i],
                label_str[i],
            )

        for predicted_text, target_text in zip(pred_str, label_str):
            pred_intent, pred_transcript = parse_multitask_output(
                predicted_text,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            true_intent, true_transcript = parse_multitask_output(
                target_text,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )

            true_label = _canonical_intent_label(true_intent)
            pred_label = _canonical_intent_label(pred_intent)

            if true_label is not None:
                true_intents.append(true_label)
                canonical_pred = pred_label if pred_label is not None else "__invalid__"
                pred_intents.append(canonical_pred)
                if canonical_pred != "__invalid__":
                    valid_pred_intent_count += 1

            if true_transcript is not None:
                true_transcripts.append(true_transcript)
                pred_transcripts.append(pred_transcript or "")

        if true_intents:
            intent_f1 = f1_score(
                true_intents,
                pred_intents,
                average="macro",
                zero_division=0.0,
            )
            intent_valid_rate = valid_pred_intent_count / len(true_intents)
        else:
            intent_f1 = 0.0
            intent_valid_rate = 0.0

        filtered_pairs = [
            (predicted, reference)
            for predicted, reference in zip(pred_transcripts, true_transcripts)
            if reference is not None and reference.strip() != ""
        ]

        if filtered_pairs:
            filtered_pred, filtered_true = zip(*filtered_pairs)
            wer = float(
                wer_metric.compute(
                    predictions=list(filtered_pred),
                    references=list(filtered_true),
                )
            )
        else:
            wer = 0.0

        logger.info(
            "EVAL SUMMARY | n_samples=%d | n_valid_intents=%d | n_invalid_preds=%d | f1=%.4f | wer=%.4f",
            len(pred_str),
            len(true_intents),
            sum(1 for label in pred_intents if label == "__invalid__"),
            float(intent_f1),
            wer,
        )

        return {
            "intent_f1_macro": float(intent_f1),
            "intent_valid_rate": float(intent_valid_rate),
            "wer": wer,
        }

    return compute_metrics
