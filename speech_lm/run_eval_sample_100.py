"""
Evaluate the Speech LM scam detector on sample_100_balanced.

Runs inference on all audios listed in the dataset manifest, ignores unloadable
files, and prints accuracy, precision, recall, F1, and execution time stats
(including average time per minute of audio).
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

# Reduce noisy warnings (same as main.py)
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message=".*sampling_rate.*WhisperFeatureExtractor.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*Sliding Window Attention.*sdpa.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*PySoundFile failed.*audioread.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)

_speech_lm_root = Path(__file__).resolve().parent
if str(_speech_lm_root) not in sys.path:
    sys.path.insert(0, str(_speech_lm_root))

from src.pipeline import run_pipeline


def _resolve_audio_path(path_str: str, dataset_dir: Path, label: str) -> Path | None:
    """Return an existing path for this manifest row, or None if not found."""
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = dataset_dir / p
        if candidate.exists():
            return candidate
    candidate = dataset_dir / label / p.name
    if candidate.exists():
        return candidate
    return None


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest CSV with columns path, label, (optional) original."""
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "path" in row and "label" in row:
                rows.append(row)
    return rows


def _compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Binary (scam=positive) metrics. Handles edge cases (no positives/negatives)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Speech LM on sample_100_balanced"
    )
    parser.add_argument(
        "--manifest",
        "-m",
        type=Path,
        default=None,
        help="Path to manifest.csv (default: <dataset_dir>/manifest.csv)",
    )
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=Path,
        default=None,
        help="Dataset root containing manifest and scam/ non_scam/ (default: ../sample_100_balanced from speech_lm)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=Path,
        default=None,
        help="Path to system prompt",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = _speech_lm_root.parent / "sample_100_balanced"
    dataset_dir = Path(dataset_dir)

    manifest_path = args.manifest or dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    rows = _load_manifest(manifest_path)
    if not rows:
        print("Error: No rows in manifest.", file=sys.stderr)
        return 1

    # Resolve paths and filter to existing files
    entries: list[tuple[Path, bool]] = []
    missing = 0
    for row in rows:
        path = _resolve_audio_path(
            row["path"].strip(),
            dataset_dir,
            row["label"].strip(),
        )
        if path is None:
            missing += 1
            continue
        label = row["label"].strip().lower()
        is_scam_gt = label == "scam"
        entries.append((path, is_scam_gt))

    if missing > 0:
        print(f"Note: {missing} manifest paths not found on disk (skipped from list).", file=sys.stderr)

    if not entries:
        print("Error: No audio paths could be resolved.", file=sys.stderr)
        return 1

    print(f"Evaluating on {len(entries)} audio files...", file=sys.stderr)

    y_true: list[bool] = []
    y_pred: list[bool] = []
    skipped_count = 0
    total_inference_sec = 0.0
    total_audio_min = 0.0
    times_per_min: list[float] = []

    for i, (audio_path, is_scam_gt) in enumerate(entries):
        try:
            result = run_pipeline(
                audio_path=audio_path,
                config_path=args.config,
                prompt_path=args.prompt,
            )
        except Exception as e:
            print(f"Skipped {audio_path.name}: {e}", file=sys.stderr)
            skipped_count += 1
            continue

        if result.get("skipped"):
            skipped_count += 1
            print(f"Skipped (unloadable): {audio_path.name}", file=sys.stderr)
            continue

        pred = bool(result.get("is_scam", False))
        y_true.append(is_scam_gt)
        y_pred.append(pred)

        t_sec = result.get("inference_time_sec")
        if t_sec is not None:
            total_inference_sec += t_sec
        audio_sec = result.get("audio_duration_sec")
        if audio_sec is not None and audio_sec > 0:
            total_audio_min += audio_sec / 60.0
        t_per_min = result.get("inference_time_per_min_audio_sec")
        if t_per_min is not None:
            times_per_min.append(t_per_min)

    if not y_true:
        print("Error: No files were successfully processed.", file=sys.stderr)
        return 1

    metrics = _compute_metrics(y_true, y_pred)
    avg_time_per_min = total_inference_sec / total_audio_min if total_audio_min > 0 else None
    mean_tpm = sum(times_per_min) / len(times_per_min) if times_per_min else None

    # Print experiment results
    print("\n" + "=" * 60, file=sys.stderr)
    print("EXPERIMENT RESULTS (sample_100_balanced)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print(f"\nEvaluated: {metrics['n']} files  |  Skipped: {skipped_count}  |  Missing paths: {missing}", file=sys.stderr)
    print(f"\n--- Classification (scam = positive) ---", file=sys.stderr)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}", file=sys.stderr)
    print(f"  Precision: {metrics['precision']:.4f}", file=sys.stderr)
    print(f"  Recall:    {metrics['recall']:.4f}", file=sys.stderr)
    print(f"  F1 Score:  {metrics['f1']:.4f}", file=sys.stderr)
    print(f"  (TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']})", file=sys.stderr)

    print(f"\n--- Execution time ---", file=sys.stderr)
    print(f"  Total inference time:        {total_inference_sec:.2f} s", file=sys.stderr)
    print(f"  Total audio duration:        {total_audio_min * 60:.2f} s ({total_audio_min:.2f} min)", file=sys.stderr)
    if avg_time_per_min is not None:
        print(f"  Avg inference time per min audio: {avg_time_per_min:.2f} s/min", file=sys.stderr)
    if mean_tpm is not None:
        print(f"  Mean (per-file) s/min audio:      {mean_tpm:.2f} s/min", file=sys.stderr)

    # Also output a machine-friendly summary to stdout
    out = {
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1_score": round(metrics["f1"], 4),
        "n_evaluated": metrics["n"],
        "n_skipped": skipped_count,
        "total_inference_sec": round(total_inference_sec, 2),
        "total_audio_min": round(total_audio_min, 2),
        "avg_inference_time_per_min_audio_sec": round(avg_time_per_min, 2) if avg_time_per_min is not None else None,
    }
    print("\n" + json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
