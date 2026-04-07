from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


LABEL_ORDER = ["non_scam", "scam"]


def _tokenize_for_wer(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    if " " in stripped:
        return stripped.split()
    return list(stripped)


def _edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(len(seq_a) + 1):
        dp[i][0] = i
    for j in range(len(seq_b) + 1):
        dp[0][j] = j

    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def compute_wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    total_errors = 0
    total_reference_tokens = 0
    for predicted, reference in zip(predictions, references):
        pred_tokens = _tokenize_for_wer(predicted)
        ref_tokens = _tokenize_for_wer(reference)
        total_errors += _edit_distance(ref_tokens, pred_tokens)
        total_reference_tokens += max(len(ref_tokens), 1)
    if total_reference_tokens == 0:
        return 0.0
    return float(total_errors / total_reference_tokens)


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gold = [str(row["gold_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    transcripts = [str(row.get("transcript", "")) for row in rows]
    references = [str(row.get("reference_transcript", "")) for row in rows]

    macro_f1 = float(f1_score(gold, predicted, labels=LABEL_ORDER, average="macro", zero_division=0.0))
    accuracy = float(accuracy_score(gold, predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=LABEL_ORDER,
        zero_division=0.0,
    )
    cm = confusion_matrix(gold, predicted, labels=LABEL_ORDER)

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(LABEL_ORDER)
        },
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "values": cm.tolist(),
        },
        "wer": compute_wer(predictions=transcripts, references=references),
        "avg_latency_ms": {
            "asr_ms": float(sum(float(row["latency_ms"]["asr_ms"]) for row in rows) / max(len(rows), 1)),
            "qa_ms": float(sum(float(row["latency_ms"]["qa_ms"]) for row in rows) / max(len(rows), 1)),
            "total_ms": float(sum(float(row["latency_ms"]["total_ms"]) for row in rows) / max(len(rows), 1)),
        },
        "num_examples": len(rows),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_prediction_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened_rows: List[Dict[str, Any]] = []
    for row in rows:
        label_scores = dict(row.get("label_scores", {}))
        flattened_rows.append(
            {
                "gold_label": row.get("gold_label"),
                "predicted_label": row.get("predicted_label"),
                "score_non_scam": label_scores.get("non_scam"),
                "score_scam": label_scores.get("scam"),
                "transcript": row.get("transcript"),
                "reference_transcript": row.get("reference_transcript"),
                "split_name": row.get("split_name"),
                "audio_duration_seconds": row.get("audio_duration_seconds"),
                "manifest_path": row.get("manifest_path"),
                "raw_path": row.get("raw_path"),
                "audio_path": row.get("audio_path"),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flattened_rows[0].keys()) if flattened_rows else [])
        if flattened_rows:
            writer.writeheader()
            writer.writerows(flattened_rows)
