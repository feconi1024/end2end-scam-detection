from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import WhisperProcessor

_track_root = Path(__file__).resolve().parent
_repo_root = _track_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from whisper_qa.src.config import load_config, resolve_repo_relative
from whisper_qa.src.data import create_processor, load_audio
from whisper_qa.src.model import WhisperQAModel
from whisper_qa.src.questions import QuestionBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the QA-driven Whisper scam detector.")
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_track_root / "config" / "whisper_qa_medium.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--processor_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    processor_dir = args.processor_dir
    if processor_dir is not None and processor_dir.exists():
        processor = WhisperProcessor.from_pretrained(str(processor_dir))
    else:
        processor = create_processor(config.get("model", {}))

    model = WhisperQAModel(processor=processor, config=config)
    checkpoint = args.checkpoint or (
        resolve_repo_relative(config.get("training", {}).get("output_dir", "outputs/whisper_qa"), args.config)
        / "last_checkpoint.pt"
    )
    payload = torch.load(str(checkpoint), map_location="cpu")
    model.load_checkpoint_payload(payload["model_payload"])
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()

    question_bank_path = resolve_repo_relative(config.get("questions", {}).get("bank_path"), args.config)
    question_bank = QuestionBank.from_yaml(question_bank_path)

    audio = load_audio(args.audio_path, sampling_rate=int(config.get("model", {}).get("sampling_rate", 16000)))
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=int(config.get("model", {}).get("sampling_rate", 16000)),
        return_attention_mask=False,
    )["input_features"]
    input_tensor = torch.tensor(input_features, dtype=torch.float32, device=target_device)

    with torch.no_grad():
        result = model.predict_single(input_tensor, question_bank=question_bank)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
