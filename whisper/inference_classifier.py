from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.classifier_data import build_label_mappings
from src.classifier_model import load_classifier_for_inference
from src.data_processor import create_processor, load_audio_for_inference, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Whisper encoder classifier inference on an audio file."
    )
    parser.add_argument(
        "--audio_path",
        "-a",
        type=Path,
        required=True,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_whisper_root / "config" / "whisper_classifier_config.yaml",
        help="Path to whisper_classifier_config.yaml",
    )
    parser.add_argument(
        "--model_dir",
        "-m",
        type=Path,
        required=True,
        help="Path to a saved classifier model directory.",
    )
    parser.add_argument(
        "--processor_dir",
        type=Path,
        default=None,
        help="Optional saved processor directory. Falls back to base processor config.",
    )
    return parser.parse_args()


def run_inference(
    audio_path: Path,
    config_path: Path,
    model_dir: Path,
    processor_dir: Path | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    sampling_rate = int(config.get("sampling_rate", 16000))

    if processor_dir is not None and processor_dir.exists():
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(str(processor_dir))
    else:
        processor, _ = create_processor(config)

    _, id2label = build_label_mappings(config)
    model = load_classifier_for_inference(model_dir)

    audio = load_audio_for_inference(audio_path=audio_path, sampling_rate=sampling_rate)
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
        return_tensors="pt",
    )
    input_features = inputs["input_features"]

    if torch.cuda.is_available():
        model = model.to("cuda")
        input_features = input_features.to("cuda")

    with torch.no_grad():
        outputs = model(input_features=input_features)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

    predicted_id = int(probabilities.argmax().item())
    predicted_label = id2label[predicted_id]
    probability_map = {
        id2label[idx]: float(probabilities[idx].item())
        for idx in range(probabilities.shape[0])
    }

    return {
        "intent": predicted_label,
        "intent_canonical": predicted_label,
        "is_scam": predicted_label == "scam",
        "confidence": probability_map[predicted_label],
        "probabilities": probability_map,
    }


def main() -> int:
    args = parse_args()

    if not args.audio_path.exists():
        print(f"Error: audio file not found: {args.audio_path}", file=sys.stderr)
        return 1

    if not args.model_dir.exists():
        print(f"Error: model_dir not found: {args.model_dir}", file=sys.stderr)
        return 1

    result = run_inference(
        audio_path=args.audio_path,
        config_path=args.config,
        model_dir=args.model_dir,
        processor_dir=args.processor_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
