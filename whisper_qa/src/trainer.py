from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import dump_config
from .evaluation import run_evaluation

logger = logging.getLogger(__name__)


class WhisperQATrainer:
    def __init__(
        self,
        model: Any,
        processor: Any,
        question_bank: Any,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None,
        config: Mapping[str, Any],
        output_dir: Path,
    ):
        self.model = model
        self.processor = processor
        self.question_bank = question_bank
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = dict(config)
        self.training_cfg = dict(config.get("training", {}))
        self.evaluation_cfg = dict(config.get("evaluation", {}))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.config, self.output_dir / "resolved_config.yaml")

        self.device = self._resolve_device()
        self.model.to(self.device)
        self.processor.save_pretrained(str(self.output_dir / "processor"))

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=float(self.training_cfg.get("learning_rate", 2.0e-3)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.0)),
        )

        total_update_steps = max(
            1,
            (len(self.train_dataloader) * int(self.training_cfg.get("num_train_epochs", 1)))
            // max(1, int(self.training_cfg.get("gradient_accumulation_steps", 1))),
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.training_cfg.get("warmup_steps", 0)),
            num_training_steps=total_update_steps,
        )

        self.use_bf16 = bool(self.training_cfg.get("bf16", False)) and self.device.type == "cuda"
        self.use_fp16 = bool(self.training_cfg.get("fp16", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        self.global_step = 0
        self.best_metric = float("-inf")

        trainable_report = self.model.trainable_parameter_report()
        with (self.output_dir / "trainable_params.json").open("w", encoding="utf-8") as f:
            json.dump(trainable_report, f, indent=2)
        logger.info("whisper_qa trainable params: %s", trainable_report)

    def _resolve_device(self) -> torch.device:
        requested = str(self.training_cfg.get("device", "auto")).lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("training.device='cuda' was requested, but CUDA is not available.")
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _autocast_context(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _compute_batch_loss(self, batch: Mapping[str, Any], epoch_index: int) -> tuple[torch.Tensor, Dict[str, float]]:
        input_features = batch["input_features"].to(self.device)
        asr_labels = batch["asr_labels"].to(self.device)

        with self._autocast_context():
            encoder_outputs = self.model.encode_audio(input_features)
            asr_loss = self.model.compute_asr_loss(encoder_outputs=encoder_outputs, asr_labels=asr_labels)

            qa_losses = []
            for sample_index, gold_label in enumerate(batch["labels"]):
                transcript_cache = self.model.build_transcript_cache(
                    encoder_hidden_states=encoder_outputs.last_hidden_state[sample_index : sample_index + 1],
                    transcript_cache_ids=batch["transcript_cache_ids"][sample_index].to(self.device),
                )
                sampled_questions = self.question_bank.sample_training_questions(
                    seed=int(self.training_cfg.get("seed", 42))
                    + epoch_index * 100000
                    + self.global_step * 100
                    + sample_index
                )

                label_losses = []
                for label, question in sampled_questions.items():
                    prompt_ids = self.model.tokenize_prompt(self.question_bank.format_prompt(question))
                    target_text = (
                        self.question_bank.positive_answer if label == gold_label else self.question_bank.negative_answer
                    )
                    target_ids = self.model.tokenize_answer(target_text)
                    score_result = self.model.score_answer_from_cache(
                        transcript_cache=transcript_cache,
                        prompt_ids=prompt_ids,
                        target_ids=target_ids,
                    )
                    label_losses.append(score_result["loss"])
                qa_losses.append(torch.stack(label_losses).mean())

            qa_loss = torch.stack(qa_losses).mean() if qa_losses else asr_loss.new_tensor(0.0)
            total_loss = (
                float(self.training_cfg.get("asr_loss_weight", 0.5)) * asr_loss
                + float(self.training_cfg.get("qa_loss_weight", 1.0)) * qa_loss
            )

        return total_loss, {
            "asr_loss": float(asr_loss.detach().item()),
            "qa_loss": float(qa_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

    def _optimizer_step(self) -> None:
        if self.use_fp16:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            max_norm=float(self.training_cfg.get("max_grad_norm", 1.0)),
        )
        if self.use_fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _save_checkpoint(self, name: str, epoch: int, metrics: Mapping[str, Any] | None = None) -> Path:
        checkpoint_path = self.output_dir / f"{name}.pt"
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_payload": self.model.build_checkpoint_payload(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": dict(metrics or {}),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def train(self) -> Dict[str, Any]:
        epochs = int(self.training_cfg.get("num_train_epochs", 1))
        grad_accum_steps = max(1, int(self.training_cfg.get("gradient_accumulation_steps", 1)))
        log_every_steps = max(1, int(self.training_cfg.get("log_every_steps", 20)))
        eval_each_epoch = bool(self.training_cfg.get("eval_each_epoch", True))
        save_each_epoch = bool(self.training_cfg.get("save_each_epoch", True))

        self.optimizer.zero_grad(set_to_none=True)
        history: Dict[str, Any] = {"epochs": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_losses = []
            processed_steps = 0

            for step, batch in enumerate(self.train_dataloader, start=1):
                if batch is None:
                    logger.warning("Skipping empty whisper_qa training batch at epoch=%d loader_step=%d", epoch, step)
                    continue

                processed_steps += 1
                self.global_step += 1
                loss, loss_info = self._compute_batch_loss(batch=batch, epoch_index=epoch)
                running_losses.append(loss_info)

                scaled_loss = loss / grad_accum_steps
                if self.use_fp16:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % grad_accum_steps == 0 or step == len(self.train_dataloader):
                    self._optimizer_step()

                if self.global_step % log_every_steps == 0:
                    logger.info(
                        "epoch=%d step=%d total_loss=%.4f asr_loss=%.4f qa_loss=%.4f skipped_in_batch=%d",
                        epoch,
                        self.global_step,
                        loss_info["total_loss"],
                        loss_info["asr_loss"],
                        loss_info["qa_loss"],
                        int(batch.get("num_skipped_examples", 0)),
                    )

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": running_losses[-1] if running_losses else {},
                "processed_steps": processed_steps,
            }

            if processed_steps == 0:
                logger.warning("Epoch %d completed with zero valid whisper_qa batches.", epoch)

            if eval_each_epoch and self.eval_dataloader is not None:
                eval_dir = self.output_dir / "eval" / f"epoch_{epoch:02d}"
                eval_result = run_evaluation(
                    model=self.model,
                    dataloader=self.eval_dataloader,
                    question_bank=self.question_bank,
                    output_dir=eval_dir,
                )
                epoch_record["eval"] = eval_result["metrics"]
                current_metric = float(eval_result["metrics"]["macro_f1"])
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self._save_checkpoint(name="best_checkpoint", epoch=epoch, metrics=eval_result["metrics"])
                    logger.info("Saved new best checkpoint at epoch %d with macro_f1=%.4f", epoch, current_metric)

            if save_each_epoch:
                self._save_checkpoint(name=f"checkpoint_epoch_{epoch:02d}", epoch=epoch, metrics=epoch_record.get("eval"))

            history["epochs"].append(epoch_record)

        final_checkpoint = self._save_checkpoint(name="last_checkpoint", epoch=epochs)
        history["last_checkpoint"] = str(final_checkpoint)
        return history
