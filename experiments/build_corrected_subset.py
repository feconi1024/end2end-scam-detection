from __future__ import annotations

import argparse
import csv
import json
import sys
import random
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.common_metrics import canonicalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a fixed balanced subset from corrected test manifest.')
    parser.add_argument(
        '--input_manifest',
        type=Path,
        default=Path('TeleAntiFraud-28k/corrected_manifests/test_manifest.csv'),
    )
    parser.add_argument(
        '--output_manifest',
        type=Path,
        default=Path('experiments/manifests/corrected_subset_100.csv'),
    )
    parser.add_argument('--per_label', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.input_manifest.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    by_label = {'scam': [], 'non_scam': []}
    for row in rows:
        label = canonicalize_label(row.get('label'))
        by_label[label].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in ('non_scam', 'scam'):
        candidates = list(by_label[label])
        if len(candidates) < args.per_label:
            raise SystemExit(f'Not enough rows for label {label}: need {args.per_label}, found {len(candidates)}')
        rng.shuffle(candidates)
        selected.extend(candidates[: args.per_label])

    rng.shuffle(selected)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    metadata = {
        'input_manifest': str(args.input_manifest),
        'output_manifest': str(args.output_manifest),
        'seed': int(args.seed),
        'per_label': int(args.per_label),
        'counts': {
            'non_scam': args.per_label,
            'scam': args.per_label,
        },
        'total_rows': len(selected),
    }
    meta_path = args.output_manifest.with_suffix('.json')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
