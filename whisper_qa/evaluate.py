from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_track_root = Path(__file__).resolve().parent
_repo_root = _track_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.common_metrics import build_standard_report, infer_eval_scope, infer_manifest_family, write_json, write_predictions_csv
from whisper_qa.src.config import load_config, resolve_repo_relative
from whisper_qa.src.data import (
    TeleAntiFraudManifestDataset,
    WhisperQACollator,
    create_processor,
    load_manifest_records,
)
from whisper_qa.src.evaluation import run_evaluation
from whisper_qa.src.model import WhisperQAModel
from whisper_qa.src.questions import QuestionBank



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the QA-driven Whisper scam detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_track_root / "config" / "whisper_qa_medium.yaml",
    )
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--sample_100", action="store_true")
    parser.add_argument('--output_json', type=Path, default=None)
    parser.add_argument('--predictions_csv', type=Path, default=None)
    parser.add_argument('--model_name', type=str, default='whisper_qa')
    parser.add_argument('--train_family', type=str, default=None)
    parser.add_argument('--eval_family', type=str, default=None)
    parser.add_argument('--eval_scope', type=str, default=None)
    return parser.parse_args()



def infer_train_family(config_path: Path, checkpoint: Path) -> str:
    text = f'{config_path} {checkpoint}'.lower()
    if 'hard' in text:
        return 'hard_manifests'
    if 'family' in text:
        return 'family_heldout_manifests'
    return 'corrected_manifests'



def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    if args.sample_100:
        dataset_path = _repo_root / "sample_100_balanced" / "manifest.csv"
        split_name = "sample_100_balanced"
    else:
        dataset_path = resolve_repo_relative(args.dataset_path or data_cfg.get("dataset_path"), args.config)
        split_name = args.eval_split or str(data_cfg.get("eval_split", "val"))

    processor = create_processor(config.get("model", {}))
    question_bank_path = resolve_repo_relative(config.get("questions", {}).get("bank_path"), args.config)
    question_bank = QuestionBank.from_yaml(question_bank_path)

    model = WhisperQAModel(processor=processor, config=config)
    checkpoint = args.checkpoint or (
        resolve_repo_relative(config.get("training", {}).get("output_dir", "outputs/whisper_qa"), args.config)
        / "best_checkpoint.pt"
    )
    payload = torch.load(str(checkpoint), map_location="cpu")
    model.load_checkpoint_payload(payload["model_payload"])
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)

    records = load_manifest_records(
        dataset_path=dataset_path,
        split_name=split_name,
        label_column=str(data_cfg.get("label_column", "label")),
        text_column=str(data_cfg.get("text_column", "transcript")),
        max_rows=args.max_rows,
    )
    collator = WhisperQACollator(processor=processor, config=config)
    dataloader = DataLoader(
        TeleAntiFraudManifestDataset(records),
        batch_size=int(config.get("training", {}).get("per_device_eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )

    output_root = resolve_repo_relative(config.get("evaluation", {}).get("output_dir", "outputs/whisper_qa/eval"), args.config)
    output_dir = output_root / split_name
    eval_start = time.perf_counter()
    result = run_evaluation(
        model=model,
        dataloader=dataloader,
        question_bank=question_bank,
        output_dir=output_dir,
        split_name=split_name,
    )
    if target_device.type == 'cuda':
        torch.cuda.synchronize()
    total_runtime_sec = time.perf_counter() - eval_start

    rows = result['rows']
    total_audio_seconds = sum(float(row.get('audio_duration_seconds', 0.0) or 0.0) for row in rows)
    standardized = build_standard_report(
        gold_labels=[str(row['gold_label']) for row in rows],
        predicted_labels=[str(row['predicted_label']) for row in rows],
        model_name=args.model_name,
        train_family=args.train_family or infer_train_family(args.config, checkpoint),
        eval_family=args.eval_family or infer_manifest_family(dataset_path),
        eval_scope=args.eval_scope or infer_eval_scope(dataset_path),
        total_runtime_sec=total_runtime_sec,
        total_audio_seconds=total_audio_seconds,
        n_skipped=int(result.get('n_skipped', 0)),
        metadata={
            'config': str(args.config),
            'checkpoint': str(checkpoint),
            'device': str(target_device),
            'split_name': split_name,
        },
        latency_breakdown={
            'qa_internal_total_sec': float(sum(float(row['latency_ms']['total_ms']) for row in rows) / 1000.0),
            'avg_latency_ms': result['metrics'].get('avg_latency_ms', {}),
        },
    )

    summary_path = args.output_json or (output_dir / 'summary.json')
    write_json(summary_path, standardized)
    if args.predictions_csv is not None:
        write_predictions_csv(args.predictions_csv, result['rows'])

    print(json.dumps(standardized, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
