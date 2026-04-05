from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.data_processor import create_processor, load_audio_for_inference, load_config
from src.evaluator import parse_multitask_output
from src.whislu_model import load_whislu_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WhiSLU inference CLI for scam detection from raw audio."
    )
    parser.add_argument(
        "--audio_path",
        "-a",
        type=Path,
        required=True,
        help="Path to input audio file (.wav, .mp3, .flac, etc.).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_whisper_root / "config" / "whislu_config.yaml",
        help="Path to whislu_config.yaml used during training.",
    )
    parser.add_argument(
        "--model_dir",
        "-p",
        type=Path,
        required=True,
        help="Path to the directory containing the fine-tuned WhiSLU model (e.g., outputs/whislu/model).",
    )
    parser.add_argument(
        "--processor_dir",
        type=Path,
        default=None,
        help="Optional path to a saved WhisperProcessor directory. "
        "If omitted, processor will be constructed from base model + config.",
    )
    parser.add_argument(
        "--intent_only",
        action="store_true",
        help="If set, enable early-exiting style decoding that focuses on the intent tokens (short generation).",
    )
    return parser.parse_args()


def run_inference(
    audio_path: Path,
    config_path: Path,
    model_dir: Path,
    processor_dir: Path | None = None,
    intent_only: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    model_name = config["model_name"]
    sampling_rate: int = int(config.get("sampling_rate", 16000))

    if processor_dir is not None and processor_dir.exists():
        from transformers import WhisperProcessor

        processor = WhisperProcessor.from_pretrained(str(processor_dir))
    else:
        processor, _ = create_processor(config)

    model = load_whislu_for_inference(
        model_dir=str(model_dir),
        processor=processor,
    )

    try:
        audio = load_audio_for_inference(audio_path=audio_path, sampling_rate=sampling_rate)
    except Exception as e:
        return {
            "error": str(e),
            "classification_tags": [],
            "is_scam": None,
            "transcript": "",
            "raw_decoded": "",
        }

    input_features = processor.feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )["input_features"]

    input_tensor = torch.tensor(input_features, dtype=model.dtype)

    if torch.cuda.is_available():
        model = model.to("cuda")
        input_tensor = input_tensor.to("cuda")

    gen_cfg = config.get("inference", {})
    max_new_tokens = int(
        gen_cfg.get(
            "intent_only_max_new_tokens" if intent_only else "max_new_tokens",
            128 if not intent_only else 8,
        )
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_tensor,
            max_new_tokens=max_new_tokens,
            forced_decoder_ids=None,
        )

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    bos = processor.tokenizer.bos_token or "<|startoftranscript|>"
    eos = processor.tokenizer.eos_token or "<|endoftext|>"

    intent, transcript = parse_multitask_output(
        decoded,
        special_tokens=[],
        bos_token=bos,
        eos_token=eos,
    )

    is_scam: bool | None
    if intent is None:
        is_scam = None
    else:
        lower_intent = intent.lower()
        if "scam" in lower_intent or "fraud" in lower_intent:
            is_scam = True
        elif "non_scam" in lower_intent or "not_scam" in lower_intent:
            is_scam = False
        else:
            is_scam = None

    result: Dict[str, Any] = {
        "intent": intent,
        "is_scam": is_scam,
        "transcript": transcript,
        "raw_decoded": decoded,
    }
    return result


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
        intent_only=args.intent_only,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

