from pathlib import Path
import sys

from transformers import set_seed, WhisperForConditionalGeneration

_whisper_root = Path(__file__).resolve().parent
if str(_whisper_root / "src") not in sys.path:
    sys.path.insert(0, str(_whisper_root / "src"))

from src.data_processor import create_processor, load_and_prepare_datasets, load_config
from src.evaluator import build_compute_metrics_fn
from src.trainer import create_trainer


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
    model_dir = Path(training_cfg.get("output_dir", "outputs/whislu")) / "model"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=None,  # use checkpoint dtype
        low_cpu_mem_usage=True,
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

    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_cfg=training_cfg,
        output_dir=Path(training_cfg.get("output_dir", "outputs/whislu")),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()