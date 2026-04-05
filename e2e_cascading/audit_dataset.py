from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2e_cascading.src.dataset import load_config


def resolve_manifest(repo_root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_counts(rows: Iterable[Dict[str, str]]) -> Counter:
    return Counter(row["label"] for row in rows)


def path_family(path_str: str) -> str:
    parts = Path(path_str).parts
    if len(parts) < 3:
        return "<unknown>"
    return parts[2]


def path_superfamily(path_str: str) -> str:
    family = path_family(path_str)
    if "multi-agent" in family:
        return "multi-agent"
    if "imitate" in family:
        return "imitate"
    return family


def transcript_overlap(a_rows: Iterable[Dict[str, str]], b_rows: Iterable[Dict[str, str]]) -> Tuple[int, int]:
    a_texts = Counter(
        (row.get("transcript") or "").strip()
        for row in a_rows
        if (row.get("transcript") or "").strip()
    )
    b_texts = Counter(
        (row.get("transcript") or "").strip()
        for row in b_rows
        if (row.get("transcript") or "").strip()
    )
    overlap_unique = sum(1 for text in b_texts if text in a_texts)
    overlap_rows = sum(count for text, count in b_texts.items() if text in a_texts)
    return overlap_unique, overlap_rows


def print_family_breakdown(split_name: str, rows: List[Dict[str, str]]) -> None:
    print(f"\n[{split_name}] family breakdown")
    by_label = defaultdict(Counter)
    by_superfamily = defaultdict(Counter)

    for row in rows:
        by_label[row["label"]][path_family(row["path"])] += 1
        by_superfamily[row["label"]][path_superfamily(row["path"])] += 1

    for label, counts in by_label.items():
        print(f"  label={label} families={dict(counts)}")
    for label, counts in by_superfamily.items():
        print(f"  label={label} superfamilies={dict(counts)}")


def run_transcript_baseline(
    train_rows: List[Dict[str, str]],
    eval_rows: List[Dict[str, str]],
) -> Tuple[float, float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the transcript baseline audit."
        ) from exc

    x_train = [(row.get("transcript") or "").strip() for row in train_rows]
    y_train = [1 if row["label"] == "scam" else 0 for row in train_rows]
    x_eval = [(row.get("transcript") or "").strip() for row in eval_rows]
    y_eval = [1 if row["label"] == "scam" else 0 for row in eval_rows]

    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=50000,
        ),
        LogisticRegression(max_iter=1000, n_jobs=1),
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_eval)
    return float(accuracy_score(y_eval, preds)), float(f1_score(y_eval, preds))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TeleAntiFraud manifests for leakage and split artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("e2e_cascading/config/default_config.yaml"),
        help="Path to the e2e_cascading config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = args.config.resolve()
    repo_root = config_path.parent.parent.parent

    train_manifest = resolve_manifest(repo_root, cfg["dataset"]["train_manifest"])
    val_manifest = resolve_manifest(repo_root, cfg["dataset"]["val_manifest"])
    test_manifest = resolve_manifest(repo_root, cfg["dataset"]["test_manifest"])

    train_rows = load_rows(train_manifest)
    val_rows = load_rows(val_manifest)
    test_rows = load_rows(test_manifest)

    for split_name, rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        print(f"[{split_name}] total={len(rows)} labels={dict(label_counts(rows))}")
        print_family_breakdown(split_name, rows)

    print("\n[overlap]")
    for left_name, left_rows, right_name, right_rows in [
        ("train", train_rows, "val", val_rows),
        ("train", train_rows, "test", test_rows),
        ("val", val_rows, "test", test_rows),
    ]:
        overlap_unique, overlap_rows = transcript_overlap(left_rows, right_rows)
        print(
            f"  exact transcript overlap {left_name}->{right_name}: "
            f"unique={overlap_unique} rows_in_{right_name}={overlap_rows}"
        )

    print("\n[text baseline]")
    for split_name, rows in [("val", val_rows), ("test", test_rows)]:
        acc, f1 = run_transcript_baseline(train_rows, rows)
        print(f"  transcript-only {split_name}: acc={acc:.4f} f1={f1:.4f}")

    print("\n[warning]")
    superfamily_by_label = defaultdict(set)
    for rows in (train_rows, val_rows, test_rows):
        for row in rows:
            superfamily_by_label[row["label"]].add(path_superfamily(row["path"]))
    print(
        "  Distinct generation families by label: "
        + ", ".join(
            f"{label}={sorted(families)}"
            for label, families in sorted(superfamily_by_label.items())
        )
    )
    print(
        "  If labels map cleanly onto generation families, a high audio F1 can reflect "
        "generation-style detection instead of robust scam understanding."
    )


if __name__ == "__main__":
    main()
