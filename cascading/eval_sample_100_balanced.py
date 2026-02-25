from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config_loader import load_settings
from src.pipeline import ScamDetectionPipeline, PipelineResult


@dataclass
class EvalSample:
    path: Path
    label_is_scam: bool
    result: PipelineResult


def iter_audio_files(
    root_dir: Path,
    scam_dir_name: str = "scam",
    non_scam_dir_name: str = "non_scam",
    exts: Iterable[str] = (".wav", ".mp3", ".m4a"),
) -> Iterable[tuple[Path, bool]]:
    """
    Yield (audio_path, label_is_scam) pairs from the sample_100_balanced structure.

    The ground truth is inferred from directory names:
      - any path containing a directory named `scam_dir_name` -> label_is_scam=True
      - any path containing `non_scam_dir_name` -> label_is_scam=False
    """
    exts_lower = {e.lower() for e in exts}
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts_lower:
            continue

        parts = {p.name.lower() for p in path.parents}
        if scam_dir_name.lower() in parts:
            label = True
        elif non_scam_dir_name.lower() in parts:
            label = False
        else:
            # Unknown label; skip
            continue

        yield path, label


def compute_metrics(samples: list[EvalSample]) -> None:
    total_seen = len(samples)
    if total_seen == 0:
        print("No valid samples to evaluate.")
        return

    tp = fp = tn = fn = 0
    asr_sum = llm_sum = total_sum = audio_dur_sum = 0.0

    used_count = 0

    for s in samples:
        analysis = s.result.analysis
        # Ignore Unknown results
        if str(analysis.category).lower() == "unknown":
            continue

        used_count += 1

        y_true = s.label_is_scam
        y_pred = bool(analysis.is_scam)

        if y_true and y_pred:
            tp += 1
        elif not y_true and not y_pred:
            tn += 1
        elif not y_true and y_pred:
            fp += 1
        elif y_true and not y_pred:
            fn += 1

        asr_sum += s.result.asr_time_s
        llm_sum += s.result.llm_time_s
        total_sum += s.result.total_time_s
        audio_dur_sum += s.result.audio_duration_s

    print("\n=== Evaluation on sample_100_balanced ===")
    print(f"Total evaluated audio files   : {total_seen}")
    print(f"Used (non-UNKNOWN) samples    : {used_count}")
    print(f"Skipped (UNKNOWN) samples     : {total_seen - used_count}")

    if used_count == 0:
        print("No non-UNKNOWN samples to compute metrics.")
        return

    # Classification metrics (scam is positive class)
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\nClassification metrics (positive class = SCAM):")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 score : {f1:.4f}")

    # Timing metrics
    avg_asr = asr_sum / used_count
    avg_llm = llm_sum / used_count
    avg_total = total_sum / used_count
    avg_audio_dur = audio_dur_sum / used_count if used_count > 0 else 0.0

    print("\nTiming (averages over used samples):")
    print(f"  Avg audio duration      : {avg_audio_dur:.2f} s")
    print(f"  Avg ASR time            : {avg_asr:.2f} s")
    print(f"  Avg LLM time            : {avg_llm:.2f} s")
    print(f"  Avg total pipeline time : {avg_total:.2f} s")

    if audio_dur_sum > 0:
        minutes = audio_dur_sum / 60.0
        print("\nTiming normalized by audio duration (all used samples):")
        print(f"  ASR time per min audio   : {asr_sum / minutes:.2f} s / min")
        print(f"  LLM time per min audio   : {llm_sum / minutes:.2f} s / min")
        print(f"  Total time per min audio : {total_sum / minutes:.2f} s / min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cascading scam detector on sample_100_balanced dataset.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../sample_100_balanced",
        help="Root directory of sample_100_balanced (default: ../sample_100_balanced).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to cascading settings YAML (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--scam_dir_name",
        type=str,
        default="scam",
        help="Directory name that denotes scam calls (default: 'scam').",
    )
    parser.add_argument(
        "--non_scam_dir_name",
        type=str,
        default="non_scam",
        help="Directory name that denotes non-scam calls (default: 'non_scam').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).resolve()

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    cfg = load_settings(args.config)
    pipeline = ScamDetectionPipeline(cfg)

    samples: list[EvalSample] = []

    for audio_path, label_is_scam in iter_audio_files(
        root,
        scam_dir_name=args.scam_dir_name,
        non_scam_dir_name=args.non_scam_dir_name,
    ):
        print(f"Processing: {audio_path}")
        try:
            result = pipeline.run(audio_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error processing {audio_path}: {exc}")
            continue

        samples.append(EvalSample(path=audio_path, label_is_scam=label_is_scam, result=result))

    compute_metrics(samples)


if __name__ == "__main__":
    main()

