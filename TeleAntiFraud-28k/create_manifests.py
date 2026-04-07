#!/usr/bin/env python3
"""
Create train/val/test CSV manifests for TeleAntiFraud-28k.

Core output columns remain compatible with e2e_cascading:
    path,label,transcript

Additional metadata columns are included for leakage auditing and harder splits:
    family,strategy,synthesis_tag,config_path,transcript_hash,duration_seconds,
    label_source,split_mode,group_id,hard_score

Split modes:
    - stratified: example-level stratified split by (label, strategy)
    - grouped: keep identical transcript groups together across splits
    - family_heldout: keep top-level audio families disjoint across splits
    - hard_bigram: grouped split where the test set is biased toward transcript
      groups containing rare character bigrams, adapted from the paper's
      unseen-bigram hard-test idea for Chinese transcripts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


@dataclass
class TranscriptGroup:
    group_id: str
    label: str
    strategy: str
    examples: List[AudioExample]
    transcript: str
    char_bigrams: Set[str]
    hard_score: int = 0


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("*", "").replace("#", "")
    return " ".join(text.split())


def _normalized_transcript(text: str) -> str:
    return "".join(text.split())


def _char_bigrams(text: str) -> Set[str]:
    normalized = _normalized_transcript(text)
    return {normalized[i : i + 2] for i in range(max(0, len(normalized) - 1))}


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

    if json_cfg.is_file():
        try:
            with json_cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _extract_from_data(data)
            config_path = json_cfg
        except Exception:
            pass
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
        transcript_hash = hashlib.md5(_normalized_transcript(transcript).encode("utf-8")).hexdigest()

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


def _build_groups(examples: Sequence[AudioExample]) -> List[TranscriptGroup]:
    grouped: Dict[str, List[AudioExample]] = defaultdict(list)
    for ex in examples:
        group_id = ex.transcript_hash if ex.transcript else ex.rel_path.as_posix()
        grouped[group_id].append(ex)

    groups: List[TranscriptGroup] = []
    for group_id, items in grouped.items():
        labels = {ex.label for ex in items}
        strategies = {ex.strategy for ex in items}
        if len(labels) != 1 or len(strategies) != 1:
            # Keep the group intact, but surface the mixed metadata in the key.
            label = ",".join(sorted(labels))
            strategy = ",".join(sorted(strategies))
        else:
            label = items[0].label
            strategy = items[0].strategy

        transcript = max((ex.transcript for ex in items), key=len, default="")
        groups.append(
            TranscriptGroup(
                group_id=group_id,
                label=label,
                strategy=strategy,
                examples=list(items),
                transcript=transcript,
                char_bigrams=_char_bigrams(transcript),
            )
        )
    return groups


def _split_group_list(
    groups: List[TranscriptGroup],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[TranscriptGroup], List[TranscriptGroup], List[TranscriptGroup]]:
    n = len(groups)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        overflow = n_train + n_val - n
        reduce_val = min(overflow, n_val)
        n_val -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            n_train -= min(overflow, n_train)

    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]
    return train_groups, val_groups, test_groups


def stratified_split(
    examples: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample], Dict[str, object]]:
    grouped_examples: Dict[Tuple[str, str], List[AudioExample]] = defaultdict(list)
    for ex in examples:
        grouped_examples[(ex.label, ex.strategy)].append(ex)

    train: List[AudioExample] = []
    val: List[AudioExample] = []
    test: List[AudioExample] = []
    stats: Dict[str, object] = {"split_mode": "stratified", "grouped": False}

    for key, items in grouped_examples.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            overflow = n_train + n_val - n
            reduce_val = min(overflow, n_val)
            n_val -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                n_train -= min(overflow, n_train)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test, stats


def grouped_split(
    examples: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample], Dict[str, object]]:
    groups = _build_groups(examples)
    strata: Dict[Tuple[str, str], List[TranscriptGroup]] = defaultdict(list)
    for group in groups:
        strata[(group.label, group.strategy)].append(group)

    train_groups: List[TranscriptGroup] = []
    val_groups: List[TranscriptGroup] = []
    test_groups: List[TranscriptGroup] = []

    for key, items in strata.items():
        random.shuffle(items)
        g_train, g_val, g_test = _split_group_list(items, train_ratio, val_ratio)
        train_groups.extend(g_train)
        val_groups.extend(g_val)
        test_groups.extend(g_test)

    train = [ex for group in train_groups for ex in group.examples]
    val = [ex for group in val_groups for ex in group.examples]
    test = [ex for group in test_groups for ex in group.examples]
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test, {
        "split_mode": "grouped",
        "grouped": True,
        "num_groups": len(groups),
    }


def _split_family_list(
    families: List[str],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    n = len(families)
    if n == 0:
        return [], [], []
    if n == 1:
        return list(families), [], []
    if n == 2:
        return [families[0]], [], [families[1]]

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_val <= 0:
        n_val = 1
        n_train = max(1, n_train - 1)
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < n:
        n_train += 1

    train_families = families[:n_train]
    val_families = families[n_train : n_train + n_val]
    test_families = families[n_train + n_val : n_train + n_val + n_test]
    return train_families, val_families, test_families


def family_heldout_split(
    examples: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample], Dict[str, object]]:
    family_examples: Dict[str, List[AudioExample]] = defaultdict(list)
    family_label: Dict[str, str] = {}
    family_strategy: Dict[str, str] = {}

    for ex in examples:
        family_examples[ex.family].append(ex)
        family_label.setdefault(ex.family, ex.label)
        family_strategy.setdefault(ex.family, ex.strategy)

    by_label: Dict[str, List[str]] = defaultdict(list)
    for family, label in family_label.items():
        by_label[label].append(family)

    train_families: List[str] = []
    val_families: List[str] = []
    test_families: List[str] = []

    for label, families in by_label.items():
        random.shuffle(families)
        label_train, label_val, label_test = _split_family_list(
            families,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        train_families.extend(label_train)
        val_families.extend(label_val)
        test_families.extend(label_test)

    train_family_set = set(train_families)
    val_family_set = set(val_families)
    test_family_set = set(test_families)

    train = [ex for ex in examples if ex.family in train_family_set]
    val = [ex for ex in examples if ex.family in val_family_set]
    test = [ex for ex in examples if ex.family in test_family_set]
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test, {
        "split_mode": "family_heldout",
        "grouped": True,
        "group_unit": "family",
        "num_families": len(family_examples),
        "train_families": sorted(train_families),
        "val_families": sorted(val_families),
        "test_families": sorted(test_families),
    }


def hard_bigram_split(
    examples: List[AudioExample],
    train_ratio: float,
    val_ratio: float,
    hard_bigram_max_df: int,
) -> Tuple[List[AudioExample], List[AudioExample], List[AudioExample], Dict[str, object]]:
    groups = _build_groups(examples)
    bigram_df: Counter[str] = Counter()
    for group in groups:
        for bigram in group.char_bigrams:
            bigram_df[bigram] += 1

    for group in groups:
        group.hard_score = sum(
            1 for bigram in group.char_bigrams if bigram_df[bigram] <= int(hard_bigram_max_df)
        )

    strata: Dict[Tuple[str, str], List[TranscriptGroup]] = defaultdict(list)
    for group in groups:
        strata[(group.label, group.strategy)].append(group)

    train_groups: List[TranscriptGroup] = []
    val_groups: List[TranscriptGroup] = []
    test_groups: List[TranscriptGroup] = []

    for key, items in strata.items():
        random.shuffle(items)
        # Hard groups first for the test split, then random split of the remainder.
        hard_sorted = sorted(items, key=lambda g: (g.hard_score, len(g.examples)), reverse=True)
        n_total = len(hard_sorted)
        n_test = max(1, int(round(n_total * (1.0 - train_ratio - val_ratio))))
        chosen_test = hard_sorted[:n_test]
        remaining = hard_sorted[n_test:]
        random.shuffle(remaining)

        n_val = int(round(n_total * val_ratio))
        chosen_val = remaining[:n_val]
        chosen_train = remaining[n_val:]

        train_groups.extend(chosen_train)
        val_groups.extend(chosen_val)
        test_groups.extend(chosen_test)

    train = [ex for group in train_groups for ex in group.examples]
    val = [ex for group in val_groups for ex in group.examples]
    test = [ex for group in test_groups for ex in group.examples]
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_val_bigrams = set()
    for group in train_groups + val_groups:
        train_val_bigrams.update(group.char_bigrams)
    unseen_test_groups = sum(1 for group in test_groups if any(bg not in train_val_bigrams for bg in group.char_bigrams))

    return train, val, test, {
        "split_mode": "hard_bigram",
        "grouped": True,
        "num_groups": len(groups),
        "hard_bigram_max_df": int(hard_bigram_max_df),
        "test_groups_with_unseen_bigram": unseen_test_groups,
        "num_test_groups": len(test_groups),
    }


def summarize_split(
    train: Sequence[AudioExample],
    val: Sequence[AudioExample],
    test: Sequence[AudioExample],
) -> Dict[str, object]:
    def _label_counts(items: Sequence[AudioExample]) -> Dict[str, int]:
        return dict(Counter(ex.label for ex in items))

    def _strategy_counts(items: Sequence[AudioExample]) -> Dict[str, int]:
        return dict(Counter(f"{ex.label}|{ex.strategy}" for ex in items))

    def _exact_transcript_overlap(
        left: Sequence[AudioExample],
        right: Sequence[AudioExample],
    ) -> int:
        left_hashes = {ex.transcript_hash for ex in left if ex.transcript}
        right_hashes = {ex.transcript_hash for ex in right if ex.transcript}
        return len(left_hashes & right_hashes)

    def _unseen_bigram_examples(
        train_like: Sequence[AudioExample],
        eval_like: Sequence[AudioExample],
    ) -> int:
        train_bigrams: Set[str] = set()
        for ex in train_like:
            train_bigrams.update(_char_bigrams(ex.transcript))
        return sum(1 for ex in eval_like if any(bg not in train_bigrams for bg in _char_bigrams(ex.transcript)))

    def _family_overlap(
        left: Sequence[AudioExample],
        right: Sequence[AudioExample],
    ) -> int:
        left_families = {ex.family for ex in left}
        right_families = {ex.family for ex in right}
        return len(left_families & right_families)

    return {
        "train": {
            "size": len(train),
            "label_counts": _label_counts(train),
            "label_strategy_counts": _strategy_counts(train),
        },
        "val": {
            "size": len(val),
            "label_counts": _label_counts(val),
            "label_strategy_counts": _strategy_counts(val),
            "examples_with_unseen_bigram_vs_train": _unseen_bigram_examples(train, val),
        },
        "test": {
            "size": len(test),
            "label_counts": _label_counts(test),
            "label_strategy_counts": _strategy_counts(test),
            "examples_with_unseen_bigram_vs_train_val": _unseen_bigram_examples(list(train) + list(val), test),
        },
        "overlap": {
            "train_val_transcript_hash_overlap": _exact_transcript_overlap(train, val),
            "train_test_transcript_hash_overlap": _exact_transcript_overlap(train, test),
            "val_test_transcript_hash_overlap": _exact_transcript_overlap(val, test),
            "train_val_family_overlap": _family_overlap(train, val),
            "train_test_family_overlap": _family_overlap(train, test),
            "val_test_family_overlap": _family_overlap(val, test),
        },
    }


def write_manifest(
    path: Path,
    examples: Sequence[AudioExample],
    split_mode: str,
    group_scores: Dict[str, int],
) -> None:
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
                "split_mode",
                "group_id",
                "hard_score",
            ],
        )
        writer.writeheader()
        for ex in examples:
            group_id = ex.transcript_hash if ex.transcript else ex.rel_path.as_posix()
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
                    "split_mode": split_mode,
                    "group_id": group_id,
                    "hard_score": group_scores.get(group_id, 0),
                }
            )
    print(f"Wrote {len(examples)} entries to {path}")


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote split summary to {path}")


def _print_basic_stats(examples: Sequence[AudioExample], label_mode: str) -> None:
    label_counts = Counter(ex.label for ex in examples)
    strategy_counts = Counter((ex.label, ex.strategy) for ex in examples)
    print(
        f"Found {len(examples)} audio files with labels {dict(label_counts)} "
        f"using label_mode={label_mode}."
    )
    for key in sorted(strategy_counts):
        print(f"  {key}: {strategy_counts[key]}")


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
        help="Directory to write manifest CSVs. Default: same as --dataset-root.",
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
    parser.add_argument(
        "--split-mode",
        type=str,
        default="grouped",
        choices=["stratified", "grouped", "family_heldout", "hard_bigram"],
        help=(
            "Split strategy. `grouped` prevents identical transcripts from crossing splits. "
            "`family_heldout` keeps top-level audio families disjoint across splits. "
            "`hard_bigram` additionally biases the test set toward transcript groups with rare character bigrams."
        ),
    )
    parser.add_argument(
        "--hard-bigram-max-df",
        type=int,
        default=2,
        help="Maximum transcript-group document frequency for a character bigram to count as hard in hard_bigram mode.",
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
    _print_basic_stats(examples, args.label_mode)
    if skipped.get("unknown_label", 0):
        print(f"Skipped {skipped['unknown_label']} files with unknown labels.")

    if args.split_mode == "stratified":
        train, val, test, split_stats = stratified_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        groups = _build_groups(examples)
    elif args.split_mode == "grouped":
        train, val, test, split_stats = grouped_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        groups = _build_groups(examples)
    elif args.split_mode == "family_heldout":
        train, val, test, split_stats = family_heldout_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        groups = _build_groups(examples)
    else:
        train, val, test, split_stats = hard_bigram_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            hard_bigram_max_df=args.hard_bigram_max_df,
        )
        groups = _build_groups(examples)
        # Recompute the scores for manifest writing.
        bigram_df: Counter[str] = Counter()
        for group in groups:
            for bigram in group.char_bigrams:
                bigram_df[bigram] += 1
        for group in groups:
            group.hard_score = sum(
                1 for bigram in group.char_bigrams if bigram_df[bigram] <= int(args.hard_bigram_max_df)
            )

    group_scores = {group.group_id: group.hard_score for group in groups}
    summary = summarize_split(train, val, test)
    summary.update(split_stats)
    summary["label_mode"] = args.label_mode
    summary["seed"] = args.seed

    write_manifest(args.output_dir / "train_manifest.csv", train, args.split_mode, group_scores)
    write_manifest(args.output_dir / "val_manifest.csv", val, args.split_mode, group_scores)
    write_manifest(args.output_dir / "test_manifest.csv", test, args.split_mode, group_scores)
    write_summary(args.output_dir / "split_summary.json", summary)


if __name__ == "__main__":
    main()
