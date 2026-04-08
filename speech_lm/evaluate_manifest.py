from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_speech_lm_root = Path(__file__).resolve().parent
_repo_root = _speech_lm_root.parent
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
from speech_lm.src.pipeline import run_pipeline
from speech_lm.src.slm_engine import SpeechLanguageModel



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Speech LM baseline on a manifest CSV.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--prompt', type=Path, default=None)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--output_json', type=Path, default=None)
    parser.add_argument('--predictions_csv', type=Path, default=None)
    parser.add_argument('--model_name', type=str, default='speech_lm')
    parser.add_argument('--train_family', type=str, default='not_applicable')
    parser.add_argument('--eval_family', type=str, default=None)
    parser.add_argument('--eval_scope', type=str, default=None)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    rows = load_manifest_rows(args.manifest, max_rows=args.max_rows)
    slm = SpeechLanguageModel(config_path=args.config)

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    prediction_rows: list[dict[str, object]] = []
    n_skipped = 0
    total_audio_seconds = 0.0
    model_runtime_total = 0.0

    eval_start = time.perf_counter()
    for row in rows:
        audio_path = resolve_manifest_audio_path(row['path'], args.manifest)
        result = run_pipeline(
            audio_path=audio_path,
            config_path=args.config,
            prompt_path=args.prompt,
            slm=slm,
        )
        if result.get('skipped') or result.get('parse_error') or 'is_scam' not in result:
            n_skipped += 1
            continue

        gold = canonicalize_label(row.get('label'))
        pred = canonicalize_label(bool(result.get('is_scam', False)))
        gold_labels.append(gold)
        predicted_labels.append(pred)
        audio_duration_seconds = float(result.get('audio_duration_sec', 0.0) or 0.0)
        total_audio_seconds += audio_duration_seconds
        model_runtime_total += float(result.get('inference_time_sec', 0.0) or 0.0)
        prediction_rows.append(
            {
                'audio_path': str(audio_path),
                'gold_label': gold,
                'predicted_label': pred,
                'audio_duration_seconds': audio_duration_seconds,
                'inference_time_sec': float(result.get('inference_time_sec', 0.0) or 0.0),
                'fraud_type': result.get('fraud_type'),
                'confidence_score': result.get('confidence_score'),
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
            'config': str(args.config) if args.config else None,
            'prompt': str(args.prompt) if args.prompt else None,
        },
        latency_breakdown={
            'model_runtime_total_sec': model_runtime_total,
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
