from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABEL_ORDER = ("non_scam", "scam")
REPO_ROOT = Path(__file__).resolve().parents[1]


def canonicalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "scam" if value else "non_scam"

    if value is None:
        raise ValueError("Label is None.")

    text = str(value).strip().lower()
    if text in {"non_scam", "non scam", "ham", "pos", "0", "false", "legitimate", "safe"}:
        return "non_scam"
    if text in {"scam", "fraud", "neg", "1", "true"}:
        return "scam"
    raise ValueError(f"Unsupported label value: {value}")


def load_manifest_rows(manifest_path: str | Path, max_rows: int | None = None) -> list[dict[str, str]]:
    manifest = Path(manifest_path)
    rows: list[dict[str, str]] = []
    with manifest.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(dict(row))
            if max_rows is not None and idx + 1 >= max_rows:
                break
    return rows


def resolve_manifest_audio_path(path_str: str, manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path).resolve()
    raw = Path(path_str)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                (manifest.parent / raw),
                (manifest.parent.parent / raw),
                (REPO_ROOT / raw),
                (REPO_ROOT / 'TeleAntiFraud-28k' / raw),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[-1].resolve() if candidates else raw.resolve()


def infer_manifest_family(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    parent = path.parent.name
    if parent.endswith('_manifests'):
        return parent
    stem = path.stem.lower()
    if 'corrected' in stem:
        return 'corrected_manifests'
    if 'family' in stem:
        return 'family_heldout_manifests'
    if 'hard' in stem:
        return 'hard_manifests'
    return parent or 'unknown'


def infer_eval_scope(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    stem = path.stem.lower()
    family = infer_manifest_family(path)
    if 'subset' in stem:
        return path.stem
    if family == 'corrected_manifests' and path.name == 'test_manifest.csv':
        return 'full_corrected_test'
    return path.stem


def build_standard_report(
    *,
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
    model_name: str,
    train_family: str,
    eval_family: str,
    eval_scope: str,
    total_runtime_sec: float,
    total_audio_seconds: float,
    n_skipped: int = 0,
    metadata: Mapping[str, Any] | None = None,
    latency_breakdown: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    gold = [canonicalize_label(v) for v in gold_labels]
    pred = [canonicalize_label(v) for v in predicted_labels]

    if len(gold) != len(pred):
        raise ValueError(f'gold/pred length mismatch: {len(gold)} vs {len(pred)}')
    if not gold:
        raise ValueError('Cannot build report with zero evaluated examples.')

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold,
        pred,
        labels=list(LABEL_ORDER),
        average='macro',
        zero_division=0.0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        pred,
        labels=list(LABEL_ORDER),
        zero_division=0.0,
    )
    cm = confusion_matrix(gold, pred, labels=list(LABEL_ORDER))
    total_audio_min = float(total_audio_seconds / 60.0) if total_audio_seconds > 0 else 0.0
    latency = float(total_runtime_sec / total_audio_min) if total_audio_min > 0 else None

    report: Dict[str, Any] = {
        'model_name': str(model_name),
        'train_family': str(train_family),
        'eval_family': str(eval_family),
        'eval_scope': str(eval_scope),
        'n_examples': int(len(gold)),
        'n_skipped': int(n_skipped),
        'accuracy': float(accuracy_score(gold, pred)),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'per_class': {
            label: {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx]),
            }
            for idx, label in enumerate(LABEL_ORDER)
        },
        'confusion_matrix': {
            'labels': list(LABEL_ORDER),
            'values': cm.tolist(),
        },
        'total_runtime_sec': float(total_runtime_sec),
        'total_audio_min': float(total_audio_min),
        'latency_sec_per_min_audio': latency,
    }
    if metadata:
        report['metadata'] = dict(metadata)
    if latency_breakdown:
        report['latency_breakdown'] = dict(latency_breakdown)
    return report


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)



def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")



def write_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flattened = [dict(row) for row in rows]
    if not flattened:
        target.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    for row in flattened:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
