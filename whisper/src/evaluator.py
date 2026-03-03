from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import evaluate
import json
import numpy as np
from sklearn.metrics import f1_score
from transformers import WhisperProcessor


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
        return None, None

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

            if t_intent is not None:
                true_intents.append(t_intent)
                pred_intents.append(p_intent if p_intent is not None else "__invalid__")

            if t_text is not None:
                true_transcripts.append(t_text)
                pred_transcripts.append(p_text or "")

        # Intent F1 (single-label classification)
        if true_intents:
            intent_f1 = f1_score(
                true_intents,
                pred_intents,
                average="micro",
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

        return {
            "intent_f1_micro": float(intent_f1),
            "wer": wer,
        }

    return compute_metrics

