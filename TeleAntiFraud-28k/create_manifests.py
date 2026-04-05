#!/usr/bin/env python3
"""
Create train/val/test CSV manifests for TeleAntiFraud-28k.

The output keeps the core columns expected by e2e_cascading:
    path,label,transcript

and also adds metadata columns that are useful for leakage audits:
    family,strategy,synthesis_tag,config_path,transcript_hash,duration_seconds,label_source

Important note:
the local TeleAntiFraud release uses folder families such as `NEG-imitate-10`
and `POS-multi-agent-8`. In this release, the family prefix is a much safer
source of the fraud-detection label than the filename prefix (`tts_fraud` /
`tts_test`), because both scam and non-scam examples appear across different
construction strategies while the filename prefix can reflect synthesis/output
pipeline details instead of the fraud label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCAM_LABEL = "scam"
NON_SCAM_LABEL = "non_scam"


@dataclass
class AudioExample:
    rel_path: Path
    label: str
    transcript: str
    family: str
    strategy: str
    synthesis_tag: str
    config_rel_path: Optional[Path]
    transcript_hash: str
    duration_seconds: float
    label_source: str


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


def _infer_strategy(family_name: str) -> str:
    if "multi-agent" in family_name:
        return "multi-agent"
    if "imitate" in family_name:
        return "imitate"
    return "unknown"


def _infer_label_from_family(family_name: str) -> str:
    if family_name.startswith("NEG-"):
        return SCAM_LABEL
    if family_name.startswith("POS-"):
        return NON_SCAM_LABEL
    raise ValueError(f"Unable to infer label from family prefix: {family_name}")


def _infer_label_from_filename(path_str: str) -> str:
    if "tts_fraud" in path_str:
        return SCAM_LABEL
    if "tts_test" in path_str:
        return NON_SCAM_LABEL
    raise ValueError(f"Unable to infer label from filename: {path_str}")


def _load_metadata_for_audio(audio_path: Path) -> Tuple[str, Optional[Path], float]:
    """
    Given an absolute path to an audio file, load a neighbouring config file
    (config.json or config.jsonl, if present) and concatenate all
    `audio_segments[*].content` fields into a single transcript string.
    """
    parent = audio_path.parent
    json_cfg = parent / "config.json"
    jsonl_cfg = parent / "config.jsonl"

    contents: List[str] = []
    config_path: Optional[Path] = None
    duration_seconds = 0.0

    def _extract_from_data(data: dict) -> None:
        nonlocal duration_seconds
        segments = data.get("audio_segments") or []
        for seg in segments:
            text = seg.get("content")
            if isinstance(text, str) and text.strip():
                contents.append(_clean_text(text))
            end_time_seconds = seg.get("end_time_seconds")
            if isinstance(end_time_seconds, (int, float)):
                duration_seconds = max(duration_seconds, float(end_time_seconds))

    # Prefer config.json if available
    if json_cfg.is_file():
        try:
            with json_cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _extract_from_data(data)
            config_path = json_cfg
        except Exception:
            pass
    # Fallback: config.jsonl (one JSON object per line)
    elif jsonl_cfg.is_file():
        try:
            with jsonl_cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _extract_from_data(data)
            config_path = jsonl_cfg
        except Exception:
            pass

    return " ".join(contents), config_path, duration_seconds


def find_audio_examples(
    dataset_root: Path,
    label_mode: str,
) -> Tuple[List[AudioExample], Dict[str, int]]:
    """
    Discover all .mp3 files under <dataset_root>/merged_result.
    """
    merged = dataset_root / "merged_result"
    if not merged.is_dir():
        raise FileNotFoundError(f"merged_result not found under {dataset_root}")

    examples: List[AudioExample] = []
    skipped: Dict[str, int] = {"unknown_label": 0}

    for mp3 in merged.rglob("*.mp3"):
        if not mp3.is_file():
            continue
        rel = mp3.relative_to(dataset_root)
        path_str = mp3.as_posix()
        family = mp3.parent.parent.name
        strategy = _infer_strategy(family)
        synthesis_tag = mp3.parent.name.split("_")[0]
        transcript, config_path, duration_seconds = _load_metadata_for_audio(mp3)
        transcript_hash = hashlib.md5(" ".join(transcript.split()).encode("utf-8")).hexdigest()

        try:
            if label_mode == "family_prefix":
                label = _infer_label_from_family(family)
                label_source = "family_prefix"
            elif label_mode == "filename":
                label = _infer_label_from_filename(path_str)
                label_source = "filename"
            else:
                raise ValueError(f"Unsupported label_mode: {label_mode}")
        except ValueError:
            skipped["unknown_label"] += 1
            continue

        config_rel_path = config_path.relative_to(dataset_root) if config_path is not None else None
        examples.append(
            AudioExample(
                rel_path=rel,
                label=label,
                transcript=transcript,
                family=family,
                strategy=strategy,
                synthesis_tag=synthesis_tag,
                config_rel_path=config_rel_path,
                transcript_hash=transcript_hash,
                duration_seconds=duration_seconds,
                label_source=label_source,
            )
        )

    return examples, skipped


def stratified_split(
    examples: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample]]:
    """
    Stratified split by (label, strategy) to avoid collapsing construction
    strategy into the target label.
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

    grouped: Dict[Tuple[str, str], List[AudioExample]] = {}
    for ex in examples:
        grouped.setdefault((ex.label, ex.strategy), []).append(ex)

    train: List[AudioExample] = []
    val: List[AudioExample] = []
    test: List[AudioExample] = []

    for key, items in grouped.items():
        random.shuffle(items)
        train_items, val_items, test_items = split_list(items)
        train.extend(train_items)
        val.extend(val_items)
        test.extend(test_items)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_manifest(path: Path, examples: List[AudioExample]) -> None:
    """
    Write a CSV manifest with core training columns plus audit metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "transcript",
                "family",
                "strategy",
                "synthesis_tag",
                "config_path",
                "transcript_hash",
                "duration_seconds",
                "label_source",
            ],
        )
        writer.writeheader()
        for ex in examples:
            writer.writerow(
                {
                    "path": "TeleAntiFraud-28k/" + ex.rel_path.as_posix(),
                    "label": ex.label,
                    "transcript": ex.transcript or "",
                    "family": ex.family,
                    "strategy": ex.strategy,
                    "synthesis_tag": ex.synthesis_tag,
                    "config_path": "" if ex.config_rel_path is None else "TeleAntiFraud-28k/" + ex.config_rel_path.as_posix(),
                    "transcript_hash": ex.transcript_hash,
                    "duration_seconds": f"{ex.duration_seconds:.3f}",
                    "label_source": ex.label_source,
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
    parser.add_argument(
        "--label-mode",
        type=str,
        default="family_prefix",
        choices=["family_prefix", "filename"],
        help=(
            "How to infer the fraud-detection label. "
            "`family_prefix` uses POS/NEG folder prefixes and is recommended for this release. "
            "`filename` preserves the previous tts_fraud/tts_test behavior."
        ),
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

    examples, skipped = find_audio_examples(args.dataset_root, label_mode=args.label_mode)
    label_counts = {"scam": 0, "non_scam": 0}
    strategy_counts: Dict[Tuple[str, str], int] = {}
    for ex in examples:
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
        key = (ex.label, ex.strategy)
        strategy_counts[key] = strategy_counts.get(key, 0) + 1
    print(
        f"Found {len(examples)} audio files with labels {label_counts} "
        f"using label_mode={args.label_mode}."
    )
    for key in sorted(strategy_counts):
        print(f"  {key}: {strategy_counts[key]}")
    if skipped.get("unknown_label", 0):
        print(f"Skipped {skipped['unknown_label']} files with unknown labels.")

    train, val, test = stratified_split(examples, args.train_ratio, args.val_ratio)

    write_manifest(args.output_dir / "train_manifest.csv", train)
    write_manifest(args.output_dir / "val_manifest.csv", val)
    write_manifest(args.output_dir / "test_manifest.csv", test)


if __name__ == "__main__":
    main()

