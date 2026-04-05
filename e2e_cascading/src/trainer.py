from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from .loss import JointCTCSLULoss


@dataclass
class TrainerConfig:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    lr_after_unfreeze_factor: float
    freeze_epochs: int
    audio_unfreeze_num_layers: int
    log_interval: int
    device: str
    output_dir: Path
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    mixed_precision: str = "bf16"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: JointCTCSLULoss,
        cfg: TrainerConfig,
        steps_per_epoch: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.mixed_precision = str(cfg.mixed_precision).lower()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.grad_scaler: Optional[torch.amp.GradScaler]
        if self.device.type == "cuda" and self.mixed_precision == "fp16":
            self.grad_scaler = torch.amp.GradScaler("cuda")
        else:
            self.grad_scaler = None

        # LR scheduler: linear warmup + cosine decay
        total_steps = steps_per_epoch * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = self._build_scheduler(warmup_steps, total_steps)

        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: freeze audio encoder
        self._freeze_audio_encoder()

    def _build_scheduler(self, warmup_steps: int, total_steps: int) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _freeze_audio_encoder(self) -> None:
        if not hasattr(self.model, "audio_encoder"):
            return
        for p in self.model.audio_encoder.parameters():
            p.requires_grad = False

    def _unfreeze_top_audio_layers(self) -> None:
        """
        Unfreeze the top-N transformer layers of the Whisper encoder.
        """
        if not hasattr(self.model, "audio_encoder"):
            return

        encoder = self.model.audio_encoder
        layers = getattr(encoder, "layers", None)
        if layers is None:
            # Fallback: unfreeze entire encoder
            for p in encoder.parameters():
                p.requires_grad = True
            return

        n = min(self.cfg.audio_unfreeze_num_layers, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Optionally unfreeze final layer norm if present
        final_ln = getattr(encoder, "layer_norm", None)
        if final_ln is not None:
            for p in final_ln.parameters():
                p.requires_grad = True

    def _maybe_switch_phase(self, epoch: int) -> None:
        """
        After freeze_epochs, unfreeze top audio layers and lower LR.
        """
        if epoch == self.cfg.freeze_epochs:
            self._unfreeze_top_audio_layers()
            # Scale down the scheduler's base_lrs so cosine decay continues
            # from the reduced LR (LambdaLR multiplies base_lr by lambda).
            factor = self.cfg.lr_after_unfreeze_factor
            self.scheduler.base_lrs = [
                lr * factor for lr in self.scheduler.base_lrs
            ]

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.mixed_precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.mixed_precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _use_grad_scaler(self) -> bool:
        return self.grad_scaler is not None

    def _forward_batch(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        input_features = batch["input_features"].to(self.device)
        audio_attention_mask = batch["audio_attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        ctc_targets = batch["ctc_targets"].to(self.device)
        ctc_target_lengths = batch["ctc_target_lengths"].to(self.device)

        with self._autocast_context():
            outputs = self.model(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
            )
            classification_logits = outputs["classification_logits"]
            ctc_logits = outputs["ctc_logits"]
            projected_attention_mask = outputs["projected_attention_mask"]

        # CTC input lengths are derived from the downsampled attention mask
        if projected_attention_mask is None:
            # Fall back to full length if mask is unavailable
            ctc_input_lengths = torch.full(
                (ctc_logits.size(0),),
                fill_value=ctc_logits.size(1),
                dtype=torch.long,
                device=ctc_logits.device,
            )
        else:
            ctc_input_lengths = projected_attention_mask.long().sum(dim=1)

        loss_dict = self.loss_fn(
            classification_logits=classification_logits.float(),
            ctc_logits=ctc_logits.float(),
            labels=labels,
            ctc_targets=ctc_targets,
            ctc_input_lengths=ctc_input_lengths,
            ctc_target_lengths=ctc_target_lengths,
        )
        return outputs, loss_dict

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> None:
        self.model.train()
        running_loss = 0.0
        accum = self.cfg.gradient_accumulation_steps

        progress = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )

        for step, batch in progress:
            _, loss_dict = self._forward_batch(batch)

            loss = loss_dict["loss"] / accum
            if self._use_grad_scaler():
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % accum == 0 or step == len(dataloader):
                if self._use_grad_scaler():
                    self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                if self._use_grad_scaler():
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss_dict["loss"].item()
            progress.set_postfix(loss=loss_dict["loss"].item())
            if step % self.cfg.log_interval == 0:
                avg_loss = running_loss / self.cfg.log_interval
                print(f"Epoch {epoch} step {step}: loss={avg_loss:.4f}")
                running_loss = 0.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        self.model.eval()

        all_labels = []
        all_preds = []

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            outputs, loss_dict = self._forward_batch(batch)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            preds = outputs["classification_logits"].argmax(dim=-1).cpu()
            labels = batch["labels"]

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

        avg_loss = total_loss / max(1, num_batches)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="binary" if len(set(all_labels)) == 2 else "macro",
            zero_division=0,
        )

        metrics = {
            f"{split_name}_loss": avg_loss,
            f"{split_name}_accuracy": float(acc),
            f"{split_name}_precision": float(precision),
            f"{split_name}_recall": float(recall),
            f"{split_name}_f1": float(f1),
        }
        print(
            f"[{split_name}] loss={avg_loss:.4f} "
            f"acc={acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}"
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        best_metric = float("-inf")
        best_ckpt_path: Optional[Path] = None
        final_ckpt_path: Optional[Path] = None

        for epoch in range(self.cfg.num_epochs):
            self._maybe_switch_phase(epoch)
            print(f"=== Epoch {epoch + 1}/{self.cfg.num_epochs} ===")
            self.train_epoch(epoch, train_loader)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, split_name="val")
                val_f1 = float(val_metrics.get("val_f1", float("-inf")))
                if val_f1 > best_metric:
                    best_metric = val_f1
                    best_ckpt_path = self.output_dir / "best_model.pt"
                    torch.save(self.model.state_dict(), best_ckpt_path)
                    print(
                        f"Saved new best checkpoint to {best_ckpt_path} "
                        f"(val_f1={val_f1:.4f})"
                    )

            # Save checkpoint after each epoch
            ckpt_path = self.output_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            final_ckpt_path = ckpt_path

        return {
            "best_checkpoint": best_ckpt_path,
            "best_val_f1": None if best_ckpt_path is None else best_metric,
            "final_checkpoint": final_ckpt_path,
        }

