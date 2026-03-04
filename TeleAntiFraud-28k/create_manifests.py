#!/usr/bin/env python3
"""
Create train/val/test CSV manifests for TeleAntiFraud-28k.

The manifests follow the format expected by e2e_cascading:
    path,label,transcript

where:
  - path: relative path to the audio file from the dataset root
  - label: "scam" or "non_scam"
  - transcript: left empty by default (no transcripts available)

We infer labels from the filename pattern under merged_result:
  - "tts_fraud" -> scam
  - "tts_test"  -> non_scam
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


SCAM_LABEL = "scam"
NON_SCAM_LABEL = "non_scam"


@dataclass
class AudioExample:
    rel_path: Path
    label: str
    transcript: str


def _clean_text(text: str) -> str:
    """
    Normalize transcript text from config files:
    - remove newlines
    - remove asterisk '*' and hash '#'
    - collapse excessive whitespace
    """
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    # Remove specific unwanted characters
    text = text.replace("*", "").replace("#", "")
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def _load_transcript_for_audio(audio_path: Path) -> str:
    """
    Given an absolute path to an audio file, load a neighbouring config file
    (config.json or config.jsonl, if present) and concatenate all
    `audio_segments[*].content` fields into a single transcript string.
    """
    parent = audio_path.parent
    json_cfg = parent / "config.json"
    jsonl_cfg = parent / "config.jsonl"

    contents: List[str] = []

    def _extract_from_data(data: dict) -> None:
        segments = data.get("audio_segments") or []
        for seg in segments:
            text = seg.get("content")
            if isinstance(text, str) and text.strip():
                contents.append(_clean_text(text))

    # Prefer config.json if available
    if json_cfg.is_file():
        try:
            with json_cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _extract_from_data(data)
        except Exception:
            pass
    # Fallback: config.jsonl (one JSON object per line)
    elif jsonl_cfg.is_file():
        try:
            with jsonl_cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _extract_from_data(data)
        except Exception:
            pass

    return " ".join(contents)


def find_audio_examples(dataset_root: Path) -> Tuple[List[AudioExample], List[AudioExample]]:
    """
    Discover all .mp3 files under <dataset_root>/merged_result and split by label.
    """
    merged = dataset_root / "merged_result"
    if not merged.is_dir():
        raise FileNotFoundError(f"merged_result not found under {dataset_root}")

    scams: List[AudioExample] = []
    non_scams: List[AudioExample] = []

    for mp3 in merged.rglob("*.mp3"):
        rel = mp3.relative_to(dataset_root)
        path_str = mp3.as_posix()
        transcript = _load_transcript_for_audio(mp3)
        if "tts_fraud" in path_str:
            scams.append(AudioExample(rel_path=rel, label=SCAM_LABEL, transcript=transcript))
        elif "tts_test" in path_str:
            non_scams.append(AudioExample(rel_path=rel, label=NON_SCAM_LABEL, transcript=transcript))
        # ignore other naming patterns if any

    return scams, non_scams


def stratified_split(
    scams: List[AudioExample],
    non_scams: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample]]:
    """
    Stratified split of scam/non-scam lists into train/val/test.
    """
    def split_list(items: List[AudioExample]) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample]]:
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure we don't exceed n due to rounding; assign remainder to test.
        if n_train + n_val > n:
            overflow = n_train + n_val - n
            # Reduce validation first, then train if needed.
            reduce_val = min(overflow, n_val)
            n_val -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                n_train -= min(overflow, n_train)
        n_test = n - n_train - n_val

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]
        return train_items, val_items, test_items

    random.shuffle(scams)
    random.shuffle(non_scams)

    scam_train, scam_val, scam_test = split_list(scams)
    non_train, non_val, non_test = split_list(non_scams)

    train = scam_train + non_train
    val = scam_val + non_val
    test = scam_test + non_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_manifest(path: Path, examples: List[AudioExample]) -> None:
    """
    Write a CSV manifest with columns: path,label,transcript
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "transcript"])
        writer.writeheader()
        for ex in examples:
            writer.writerow(
                {
                    "path": 'TeleAntiFraud-28k/' + ex.rel_path.as_posix(),
                    "label": ex.label,
                    "transcript": ex.transcript or "",
                }
            )
    print(f"Wrote {len(examples)} entries to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test CSV manifests for TeleAntiFraud-28k."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root of TeleAntiFraud-28k (folder containing merged_result). "
        "Default: directory containing this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write manifest CSVs. "
        "Default: same as --dataset-root.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1). The rest goes to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = Path(__file__).resolve().parent
    args.dataset_root = args.dataset_root.resolve()

    if args.output_dir is None:
        args.output_dir = args.dataset_root
    args.output_dir = args.output_dir.resolve()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={args.train_ratio}, val_ratio={args.val_ratio}. "
            "Require train_ratio>0, val_ratio>=0, and train_ratio+val_ratio<1."
        )

    random.seed(args.seed)

    scams, non_scams = find_audio_examples(args.dataset_root)
    print(f"Found {len(scams)} scam and {len(non_scams)} non-scam audio files.")

    train, val, test = stratified_split(scams, non_scams, args.train_ratio, args.val_ratio)

    write_manifest(args.output_dir / "train_manifest.csv", train)
    write_manifest(args.output_dir / "val_manifest.csv", val)
    write_manifest(args.output_dir / "test_manifest.csv", test)


if __name__ == "__main__":
    main()

