from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_track_root = Path(__file__).resolve().parent
_repo_root = _track_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.common_metrics import (
    build_standard_report,
    canonicalize_label,
    infer_eval_scope,
    infer_manifest_family,
    load_manifest_rows,
    resolve_manifest_audio_path,
    write_json,
    write_predictions_csv,
)
from cascading.src.config_loader import load_settings
from cascading.src.pipeline import ScamDetectionPipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate cascading baseline on a manifest CSV.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=_track_root / 'config' / 'settings.yaml')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--output_json', type=Path, default=None)
    parser.add_argument('--predictions_csv', type=Path, default=None)
    parser.add_argument('--model_name', type=str, default='asr_llm')
    parser.add_argument('--train_family', type=str, default='not_applicable')
    parser.add_argument('--eval_family', type=str, default=None)
    parser.add_argument('--eval_scope', type=str, default=None)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    rows = load_manifest_rows(args.manifest, max_rows=args.max_rows)
    cfg = load_settings(str(args.config))
    pipeline = ScamDetectionPipeline(cfg)

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    prediction_rows: list[dict[str, object]] = []
    n_skipped = 0
    total_audio_seconds = 0.0
    asr_total_sec = 0.0
    llm_total_sec = 0.0

    eval_start = time.perf_counter()
    for row in rows:
        audio_path = resolve_manifest_audio_path(row['path'], args.manifest)
        result = pipeline.run(audio_path)
        category = str(getattr(result.analysis, 'category', '')).strip().lower()
        if category == 'unknown':
            n_skipped += 1
            continue

        gold = canonicalize_label(row.get('label'))
        pred = canonicalize_label(bool(getattr(result.analysis, 'is_scam', False)))
        gold_labels.append(gold)
        predicted_labels.append(pred)
        total_audio_seconds += float(result.audio_duration_s)
        asr_total_sec += float(result.asr_time_s)
        llm_total_sec += float(result.llm_time_s)
        prediction_rows.append(
            {
                'audio_path': str(audio_path),
                'gold_label': gold,
                'predicted_label': pred,
                'audio_duration_seconds': float(result.audio_duration_s),
                'asr_time_s': float(result.asr_time_s),
                'llm_time_s': float(result.llm_time_s),
                'pipeline_time_s': float(result.total_time_s),
                'category': getattr(result.analysis, 'category', ''),
                'risk_score': getattr(result.analysis, 'risk_score', None),
            }
        )
    total_runtime_sec = time.perf_counter() - eval_start

    report = build_standard_report(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        model_name=args.model_name,
        train_family=args.train_family,
        eval_family=args.eval_family or infer_manifest_family(args.manifest),
        eval_scope=args.eval_scope or infer_eval_scope(args.manifest),
        total_runtime_sec=total_runtime_sec,
        total_audio_seconds=total_audio_seconds,
        n_skipped=n_skipped,
        metadata={
            'config': str(args.config),
            'backend': str(cfg.llm.backend),
        },
        latency_breakdown={
            'asr_total_sec': asr_total_sec,
            'llm_total_sec': llm_total_sec,
        },
    )

    if args.output_json is not None:
        write_json(args.output_json, report)
    if args.predictions_csv is not None:
        write_predictions_csv(args.predictions_csv, prediction_rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
