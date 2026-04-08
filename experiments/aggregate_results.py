from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable



def _iter_result_files(root: Path) -> Iterable[Path]:
    for path in root.rglob('*.json'):
        if path.name.endswith('.json'):
            yield path



def _looks_like_summary(payload: dict[str, Any]) -> bool:
    required = {'model_name', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'per_class'}
    return required.issubset(payload.keys())



def _flatten(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    per_class = payload.get('per_class', {})
    non_scam = per_class.get('non_scam', {})
    scam = per_class.get('scam', {})
    confusion = payload.get('confusion_matrix', {})
    values = confusion.get('values', [[None, None], [None, None]])
    row = {
        'source_path': str(source_path),
        'model_name': payload.get('model_name'),
        'train_family': payload.get('train_family'),
        'eval_family': payload.get('eval_family'),
        'eval_scope': payload.get('eval_scope'),
        'n_examples': payload.get('n_examples'),
        'n_skipped': payload.get('n_skipped'),
        'accuracy': payload.get('accuracy'),
        'precision_macro': payload.get('precision_macro'),
        'recall_macro': payload.get('recall_macro'),
        'f1_macro': payload.get('f1_macro'),
        'non_scam_precision': non_scam.get('precision'),
        'non_scam_recall': non_scam.get('recall'),
        'non_scam_f1': non_scam.get('f1'),
        'non_scam_support': non_scam.get('support'),
        'scam_precision': scam.get('precision'),
        'scam_recall': scam.get('recall'),
        'scam_f1': scam.get('f1'),
        'scam_support': scam.get('support'),
        'cm_00': values[0][0] if len(values) > 0 and len(values[0]) > 0 else None,
        'cm_01': values[0][1] if len(values) > 0 and len(values[0]) > 1 else None,
        'cm_10': values[1][0] if len(values) > 1 and len(values[1]) > 0 else None,
        'cm_11': values[1][1] if len(values) > 1 and len(values[1]) > 1 else None,
        'total_runtime_sec': payload.get('total_runtime_sec'),
        'total_audio_min': payload.get('total_audio_min'),
        'latency_sec_per_min_audio': payload.get('latency_sec_per_min_audio'),
    }
    metadata = payload.get('metadata', {}) or {}
    for key, value in metadata.items():
        row[f'metadata_{key}'] = value
    return row



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Aggregate standardized experiment result JSONs into CSV/JSON.')
    parser.add_argument('--input_root', type=Path, default=Path('experiments/results'))
    parser.add_argument('--output_csv', type=Path, default=Path('experiments/results/summary.csv'))
    parser.add_argument('--output_json', type=Path, default=Path('experiments/results/summary.json'))
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in _iter_result_files(args.input_root):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(payload, dict) and _looks_like_summary(payload):
            rows.append(_flatten(payload, path))

    rows.sort(key=lambda row: (str(row.get('eval_scope')), str(row.get('model_name')), str(row.get('train_family'))))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with args.output_csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        args.output_csv.write_text('', encoding='utf-8')

    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'num_results': len(rows), 'output_csv': str(args.output_csv), 'output_json': str(args.output_json)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
