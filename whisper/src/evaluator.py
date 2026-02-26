from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import evaluate
import numpy as np
from sklearn.metrics import f1_score
from transformers import WhisperProcessor


def parse_multitask_output(
    text: str,
    special_tokens: Sequence[str],
    bos_token: str | None,
    eos_token: str | None,
) -> Tuple[List[str], str]:
    """
    Split decoded output into (intent_tags, transcript_text).

    Assumes sequences like:
      <|startoftranscript|><|scam|><|impersonation|> hello world <|endoftext|>
    """
    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]

    if eos_token and eos_token in text:
        text = text.split(eos_token, 1)[0]

    tokens = text.split()
    special_set = set(special_tokens)

    intent_tags: List[str] = []
    transcript_tokens: List[str] = []

    for tok in tokens:
        if tok in special_set:
            intent_tags.append(tok)
        else:
            transcript_tokens.append(tok)

    transcript = " ".join(transcript_tokens).strip()
    return intent_tags, transcript


def build_compute_metrics_fn(
    processor: WhisperProcessor,
    special_tokens: Sequence[str],
    label2token: Mapping[str, str],
) -> Any:
    """
    Create a compute_metrics function for Seq2SeqTrainer that:
    - Extracts intent tags and transcripts from generated sequences.
    - Computes micro F1 over intent tags.
    - Computes WER over transcripts.
    """
    wer_metric = evaluate.load("wer")

    bos_token = processor.tokenizer.bos_token or "<|startoftranscript|>"
    eos_token = processor.tokenizer.eos_token or "<|endoftext|>"

    all_intents: List[str] = list(label2token.keys())
    token_to_index: Dict[str, int] = {
        tok: idx for idx, tok in enumerate(all_intents)
    }
    token_values = set(label2token.values())

    def _to_multilabel_vector(tokens: Sequence[str]) -> List[int]:
        vec = [0] * len(all_intents)
        for tok in tokens:
            # tok is a special fraud token; map back to canonical intent if possible
            # Reverse lookup label2token
            for label, s_tok in label2token.items():
                if tok == s_tok:
                    idx = all_intents.index(label)
                    vec[idx] = 1
                    break
        return vec

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

        pred_intents: List[List[int]] = []
        true_intents: List[List[int]] = []
        pred_transcripts: List[str] = []
        true_transcripts: List[str] = []

        for p, t in zip(pred_str, label_str):
            p_tags, p_text = parse_multitask_output(
                p,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            t_tags, t_text = parse_multitask_output(
                t,
                special_tokens=special_tokens,
                bos_token=bos_token,
                eos_token=eos_token,
            )

            pred_intents.append(_to_multilabel_vector(p_tags))
            true_intents.append(_to_multilabel_vector(t_tags))
            pred_transcripts.append(p_text)
            true_transcripts.append(t_text)

        # Intent F1 (multi-label, micro)
        intent_f1 = f1_score(
            true_intents,
            pred_intents,
            average="micro",
            zero_division=0.0,
        )

        # WER on transcripts
        wer = float(
            wer_metric.compute(
                predictions=pred_transcripts,
                references=true_transcripts,
            )
        )

        return {
            "intent_f1_micro": float(intent_f1),
            "wer": wer,
        }

    return compute_metrics

