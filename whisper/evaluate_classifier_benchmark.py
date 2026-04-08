from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

_whisper_root = Path(__file__).resolve().parent
_repo_root = _whisper_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_whisper_root / 'src') not in sys.path:
    sys.path.insert(0, str(_whisper_root / 'src'))

from experiments.common_metrics import (
    build_standard_report,
    infer_eval_scope,
    infer_manifest_family,
    write_json,
    write_predictions_csv,
)
from src.classifier_data import build_label_mappings, load_and_prepare_classification_datasets
from src.classifier_model import load_classifier_for_inference, resolve_classifier_model_dir
from src.data_processor import create_processor, load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')


class BenchmarkCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        fe = self.processor.feature_extractor
        sampling_rate = int(getattr(fe, 'sampling_rate', 16000))
        input_feats_list: List[Dict[str, Any]] = []
        labels: List[int] = []
        metadata: List[Dict[str, Any]] = []
        skipped = 0

        for feature in features:
            audio_path = feature.get('audio_path')
            try:
                if not audio_path or not Path(audio_path).exists():
                    raise FileNotFoundError(str(audio_path))
                array, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
                if array.ndim > 1:
                    array = np.mean(array, axis=0)
                array = np.asarray(array, dtype=np.float32)
                inputs = fe(array, sampling_rate=sampling_rate, return_attention_mask=False)
                input_feats_list.append({'input_features': inputs['input_features'][0]})
                labels.append(int(feature['class_label']))
                metadata.append(
                    {
                        'audio_path': str(audio_path),
                        'audio_duration_seconds': float(feature.get('audio_duration_seconds', 0.0) or 0.0),
                        'family': str(feature.get('family', '')),
                    }
                )
            except Exception:
                skipped += 1
                continue

        if not input_feats_list:
            return None

        batch = fe.pad(input_feats_list, padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        batch['metadata'] = metadata
        batch['num_skipped_examples'] = skipped
        return batch



def infer_train_family(config_path: Path, model_dir: Path) -> str:
    text = f'{config_path} {model_dir}'.lower()
    if 'family' in text:
        return 'family_heldout_manifests'
    if 'hard' in text:
        return 'hard_manifests'
    return 'corrected_manifests'



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark Whisper encoder classifier on a manifest.')
    parser.add_argument('--config', '-c', type=Path, default=_whisper_root / 'config' / 'whisper_classifier_config.yaml')
    parser.add_argument('--dataset_path', '-d', type=Path, required=True)
    parser.add_argument('--model_dir', '-m', type=Path, required=True)
    parser.add_argument('--processor_dir', type=Path, default=None)
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--num_proc', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_json', type=Path, default=None)
    parser.add_argument('--predictions_csv', type=Path, default=None)
    parser.add_argument('--model_name', type=str, default='whisper_classifier')
    parser.add_argument('--train_family', type=str, default=None)
    parser.add_argument('--eval_family', type=str, default=None)
    parser.add_argument('--eval_scope', type=str, default=None)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    resolved_model_dir = resolve_classifier_model_dir(args.model_dir)

    if args.processor_dir is not None and args.processor_dir.exists():
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(str(args.processor_dir))
    else:
        processor, _ = create_processor(config)

    _, eval_dataset, label2id, id2label, _, _ = load_and_prepare_classification_datasets(
        dataset_path=args.dataset_path,
        processor=processor,
        config=config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        num_proc=args.num_proc,
    )
    model = load_classifier_for_inference(resolved_model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    batch_size = args.batch_size or int(config.get('training', {}).get('per_device_eval_batch_size', 4))
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=BenchmarkCollator(processor),
    )

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    prediction_rows: list[dict[str, object]] = []
    n_skipped = 0
    total_audio_seconds = 0.0

    eval_start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            n_skipped += int(batch.get('num_skipped_examples', 0))
            input_features = batch['input_features'].to(device)
            outputs = model(input_features=input_features)
            if device == 'cuda':
                torch.cuda.synchronize()
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            labels = batch['labels'].detach().cpu().tolist()
            metadata = batch['metadata']
            for pred_id, label_id, meta in zip(preds, labels, metadata):
                gold = id2label[int(label_id)]
                pred = id2label[int(pred_id)]
                gold_labels.append(gold)
                predicted_labels.append(pred)
                total_audio_seconds += float(meta.get('audio_duration_seconds', 0.0) or 0.0)
                prediction_rows.append(
                    {
                        'audio_path': meta.get('audio_path'),
                        'gold_label': gold,
                        'predicted_label': pred,
                        'audio_duration_seconds': float(meta.get('audio_duration_seconds', 0.0) or 0.0),
                        'family': meta.get('family'),
                    }
                )
    if device == 'cuda':
        torch.cuda.synchronize()
    total_runtime_sec = time.perf_counter() - eval_start

    report = build_standard_report(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        model_name=args.model_name,
        train_family=args.train_family or infer_train_family(args.config, resolved_model_dir),
        eval_family=args.eval_family or infer_manifest_family(args.dataset_path),
        eval_scope=args.eval_scope or infer_eval_scope(args.dataset_path),
        total_runtime_sec=total_runtime_sec,
        total_audio_seconds=total_audio_seconds,
        n_skipped=n_skipped,
        metadata={
            'config': str(args.config),
            'resolved_model_dir': str(resolved_model_dir),
            'device': device,
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
