from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics, write_json, write_jsonl, write_prediction_csv


@torch.no_grad()
def run_evaluation(
    model: Any,
    dataloader: DataLoader,
    question_bank: Any,
    output_dir: Path | None = None,
    split_name: str | None = None,
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for batch in dataloader:
        if batch is None:
            continue

        input_features = batch["input_features"].to(model.device)
        for batch_index in range(input_features.shape[0]):
            prediction = model.predict_single(input_features[batch_index : batch_index + 1], question_bank=question_bank)
            rows.append(
                {
                    "gold_label": batch["labels"][batch_index],
                    "predicted_label": prediction["predicted_label"],
                    "label_scores": prediction["label_scores"],
                    "transcript": prediction["transcript"],
                    "reference_transcript": batch["transcript_texts"][batch_index],
                    "raw_question_scores": prediction["raw_question_scores"],
                    "latency_ms": prediction["latency_ms"],
                    "split_name": split_name or batch["split_names"][batch_index],
                    "audio_duration_seconds": batch["audio_durations"][batch_index],
                    "manifest_path": batch["manifest_paths"][batch_index],
                    "audio_path": batch["audio_paths"][batch_index],
                    "raw_path": batch["raw_paths"][batch_index],
                }
            )

    metrics = compute_metrics(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", metrics)
        write_jsonl(output_dir / "predictions.jsonl", rows)
        write_prediction_csv(output_dir / "predictions.csv", rows)

    return {
        "metrics": metrics,
        "rows": rows,
    }
