from pathlib import Path
import sys

from transformers import (
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
)

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.data_processor import (
    DataCollatorSpeechSeq2SeqWithPadding,
    create_processor,
    load_and_prepare_datasets,
    load_config,
)
from src.evaluator import build_compute_metrics_fn
from src.trainer import create_training_arguments


def main():
    config_path = _whisper_root / "config" / "whislu_config.yaml"
    dataset_path = Path("TeleAntiFraud-28k")  # adjust if needed

    config = load_config(config_path)
    processor, _ = create_processor(config)
    training_cfg = config.get("training", {})
    model_name = config["model_name"]

    train_ds, eval_ds = load_and_prepare_datasets(
        dataset_path=dataset_path,
        processor=processor,
        config=config,
        train_split="train",
        eval_split="validation",
        num_proc=None,
    )

    # Apply same eval subsampling as train.py to avoid OOM during generation.
    max_eval_samples = int(training_cfg.get("max_eval_samples", 0))
    if max_eval_samples > 0 and len(eval_ds) > max_eval_samples:
        eval_ds = eval_ds.select(range(max_eval_samples))

    # Load trained full model directly from disk.
    model_dir = (Path(training_cfg.get("output_dir", "outputs/whislu")) / "model").resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Trained model directory not found: {model_dir}")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=None,  # use checkpoint dtype
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Ensure generation config doesn't enforce language/task tokens so that
    # JSON-formatted SLU outputs can be produced and evaluated.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    compute_metrics = build_compute_metrics_fn(
        processor=processor,
        special_tokens=config.get("special_tokens", []),
        label2token=config.get("label2token", {}),
    )

    # Build eval-only training arguments
    output_dir = Path(training_cfg.get("output_dir", "outputs/whislu"))
    eval_args = create_training_arguments(training_cfg=training_cfg, output_dir=output_dir)

    base_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def eval_collator(features):
        # Use the standard collator but drop any auxiliary-only keys such as
        # intent_labels so that generation sees only the expected kwargs.
        batch = base_collator(features)
        batch.pop("intent_labels", None)
        return batch

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        data_collator=eval_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()