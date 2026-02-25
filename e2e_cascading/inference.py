from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import librosa
import torch
from transformers import WhisperProcessor, BertTokenizer

from e2e_cascading.src.dataset import load_config
from e2e_cascading.src.model import DifferentiableCascadeModel


def run_inference(
    audio_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    cfg = load_config(config_path)

    whisper_name = cfg["model"]["whisper_model_name"]
    bert_name = cfg["model"]["bert_model_name"]
    sample_rate = int(cfg["dataset"]["sample_rate"])

    processor = WhisperProcessor.from_pretrained(whisper_name)
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    # Label mapping must match training
    scam_label = cfg["dataset"]["scam_label"]
    non_scam_label = cfg["dataset"]["non_scam_label"]
    label_mapping = {non_scam_label: 0, scam_label: 1}
    id2label = {v: k for k, v in label_mapping.items()}

    num_labels = int(cfg["model"]["num_labels"])
    ctc_vocab_size = tokenizer.vocab_size
    ctc_blank_id = int(cfg["model"]["ctc_blank_token_id"])

    projector_cfg_overrides = cfg["model"].get("projector", {})

    model = DifferentiableCascadeModel(
        whisper_model_name=whisper_name,
        bert_model_name=bert_name,
        num_labels=num_labels,
        ctc_vocab_size=ctc_vocab_size,
        ctc_blank_id=ctc_blank_id,
        projector_cfg_overrides=projector_cfg_overrides,
        label2id=label_mapping,
        id2label=id2label,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device(cfg["training"]["device"])
    model.to(device)

    # Load and preprocess audio
    audio, sr = librosa.load(audio_path.as_posix(), sr=sample_rate, mono=True)
    inputs = processor(
        [audio],
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    # All frames are valid; WhisperProcessor already handles padding to
    # the expected number of frames for the encoder.
    bsz, _, t_acoustic = input_features.shape
    audio_attention_mask = torch.ones((bsz, t_acoustic), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
        )
        logits = outputs["classification_logits"][0]
        probs = logits.softmax(dim=-1)
        pred_id = int(probs.argmax().item())
        confidence = float(probs[pred_id].item())

    label_str = id2label[pred_id]
    is_scam = label_str == scam_label

    return {
        "is_scam": is_scam,
        "confidence": confidence,
        "category": label_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for differentiable cascaded scam detector.")
    parser.add_argument(
        "--audio_path",
        type=Path,
        required=True,
        help="Path to input audio file (e.g., .mp3 or .wav).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("e2e_cascading/config/default_config.yaml"),
        help="Path to YAML config used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    args = parser.parse_args()

    result = run_inference(
        audio_path=args.audio_path,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

