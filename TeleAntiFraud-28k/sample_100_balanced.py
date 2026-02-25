#!/usr/bin/env python3
"""
Sample 100 audio files from TeleAntiFraud-28k (50 scams, 50 non-scams).

Labelling format in this dataset:
- Scam (fraud): path or folder name contains "tts_fraud", e.g.:
    merged_result/POS-multi-agent-9/tts_fraud_02498/tts_fraud_02498.mp3
    merged_result/NEG-multi-agent-2/tts_fraud_00961/tts_fraud_00961.mp3
- Non-scam: path or folder name contains "tts_test", e.g.:
    merged_result/POS-imitate-6/tts_test330/tts_test330.mp3
    merged_result/NEG-imitate-10/tts_test944/tts_test944.mp3

Audio files are MP3s under <dataset_root>/merged_result/<category>/<sample_id>/<sample_id>.mp3.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


def find_audio_by_label(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Discover all .mp3 under merged_result and split by label (scam vs non-scam)."""
    merged = dataset_root / "merged_result"
    if not merged.is_dir():
        raise FileNotFoundError(f"merged_result not found under {dataset_root}")

    scams: list[Path] = []
    non_scams: list[Path] = []

    for mp3 in merged.rglob("*.mp3"):
        path_str = mp3.as_posix()
        if "tts_fraud" in path_str:
            scams.append(mp3)
        elif "tts_test" in path_str:
            non_scams.append(mp3)
        # ignore other naming patterns if any

    return scams, non_scams


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample 50 scam + 50 non-scam audio files from TeleAntiFraud-28k."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root of TeleAntiFraud-28k (folder containing merged_result). "
        "Default: directory containing this script / TeleAntiFraud-28k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sampled files (default: <dataset_root>/sample_100_balanced)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only write manifest CSV of selected paths; do not copy files",
    )
    args = parser.parse_args()

    if args.dataset_root is None:
        script_dir = Path(__file__).resolve().parent
        args.dataset_root = script_dir / "TeleAntiFraud-28k"
    args.dataset_root = args.dataset_root.resolve()

    if args.output_dir is None:
        args.output_dir = args.dataset_root / "sample_100_balanced"
    args.output_dir = args.output_dir.resolve()

    random.seed(args.seed)

    scams, non_scams = find_audio_by_label(args.dataset_root)
    print(f"Found {len(scams)} scam and {len(non_scams)} non-scam audio files.")

    n_scam = min(50, len(scams))
    n_non_scam = min(50, len(non_scams))
    if n_scam < 50 or n_non_scam < 50:
        print(f"Warning: sampling {n_scam} scams and {n_non_scam} non-scams (fewer than 50 in one class).")

    selected_scams = random.sample(scams, n_scam)
    selected_non_scams = random.sample(non_scams, n_non_scam)

    manifest_rows: list[dict] = []
    if not args.no_copy:
        (args.output_dir / "scam").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "non_scam").mkdir(parents=True, exist_ok=True)

        for p in selected_scams:
            dest = args.output_dir / "scam" / p.name
            shutil.copy2(p, dest)
            manifest_rows.append({
                "path": dest.as_posix(),
                "label": "scam",
                "original": p.relative_to(args.dataset_root).as_posix(),
            })
        for p in selected_non_scams:
            dest = args.output_dir / "non_scam" / p.name
            # avoid name clash with scam folder if same base name exists
            if dest.exists() and dest.resolve() != p.resolve():
                dest = args.output_dir / "non_scam" / f"non_scam_{p.name}"
            shutil.copy2(p, dest)
            manifest_rows.append({
                "path": dest.as_posix(),
                "label": "non_scam",
                "original": p.relative_to(args.dataset_root).as_posix(),
            })
        print(f"Copied {n_scam} scam and {n_non_scam} non_scam files to {args.output_dir}")
    else:
        for p in selected_scams:
            manifest_rows.append({
                "path": "",
                "label": "scam",
                "original": p.relative_to(args.dataset_root).as_posix(),
            })
        for p in selected_non_scams:
            manifest_rows.append({
                "path": "",
                "label": "non_scam",
                "original": p.relative_to(args.dataset_root).as_posix(),
            })
        print("No copy: manifest only.")

    manifest_path = args.output_dir / "manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "original"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
