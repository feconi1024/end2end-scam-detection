from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import compute_metrics, write_json, write_jsonl, write_prediction_csv


@torch.no_grad()
def run_evaluation(
    model: Any,
    dataloader: DataLoader,
    question_bank: Any,
    output_dir: Path | None = None,
    split_name: str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    total_skipped_examples = 0

    eval_iterator = tqdm(
        dataloader,
        total=len(dataloader),
        desc=progress_desc or f"Evaluate {split_name or ''}".strip(),
        leave=True,
        disable=not show_progress,
    )

    for batch in eval_iterator:
        if batch is None:
            eval_iterator.set_postfix(skipped_batch=True)
            continue

        skipped_in_batch = int(batch.get('num_skipped_examples', 0))
        total_skipped_examples += skipped_in_batch

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

        eval_iterator.set_postfix(
            examples=len(rows),
            skipped=total_skipped_examples,
        )

    metrics = compute_metrics(rows)
    metrics['n_skipped'] = int(total_skipped_examples)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", metrics)
        write_jsonl(output_dir / "predictions.jsonl", rows)
        write_prediction_csv(output_dir / "predictions.csv", rows)

    return {
        "metrics": metrics,
        "rows": rows,
        'n_skipped': int(total_skipped_examples),
    }
