from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput

from .data import encode_transcript_cache_ids
from .prefix_tuning import (
    WhisperPrefixTuningManager,
    count_trainable_parameters,
    freeze_base_model_parameters,
)

logger = logging.getLogger(__name__)


def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    for layer_idx, (key_tensor, value_tensor) in enumerate(zip(cache.key_cache, cache.value_cache)):
        cloned.update(key_tensor.clone(), value_tensor.clone(), layer_idx)
    return cloned


def clone_encoder_decoder_cache(cache: EncoderDecoderCache) -> EncoderDecoderCache:
    cloned = EncoderDecoderCache(
        self_attention_cache=_clone_dynamic_cache(cache.self_attention_cache),
        cross_attention_cache=_clone_dynamic_cache(cache.cross_attention_cache),
    )
    cloned.is_updated = dict(cache.is_updated)
    return cloned


def gather_target_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


class WhisperQAModel(nn.Module):
    def __init__(
        self,
        processor: Any,
        config: Mapping[str, Any],
        whisper_model: Optional[WhisperForConditionalGeneration] = None,
    ):
        super().__init__()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = dict(config)
        self.model_cfg = dict(config.get("model", {}))
        self.tuning_cfg = dict(config.get("tuning", {}))
        self.training_cfg = dict(config.get("training", {}))
        self.inference_cfg = dict(config.get("inference", {}))
        self.model_name = str(self.model_cfg["name"])
        self.tuning_mode = str(self.tuning_cfg.get("mode", "prefix")).lower()
        self.prefix_manager: WhisperPrefixTuningManager | None = None

        if whisper_model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = whisper_model

        self._configure_generation_for_asr()
        self._configure_tuning_mode()

        requested_gc = bool(self.training_cfg.get("gradient_checkpointing", False))
        if requested_gc:
            logger.warning(
                "Gradient checkpointing is disabled for whisper_qa because transcript-cache QA scoring requires use_cache=True during training."
            )
        self.model.config.use_cache = True

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _configure_generation_for_asr(self) -> None:
        language = self.model_cfg.get("language")
        task = str(self.model_cfg.get("task", "transcribe"))
        no_timestamps = bool(self.model_cfg.get("no_timestamps", True))

        if getattr(self.model, "generation_config", None) is not None:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                task=task,
                language=language,
                no_timestamps=no_timestamps,
            )
            self.model.generation_config.forced_decoder_ids = forced_decoder_ids
            self.model.config.forced_decoder_ids = forced_decoder_ids
            if hasattr(self.model.generation_config, "_from_model_config"):
                self.model.generation_config._from_model_config = False
            if hasattr(self.model.generation_config, "language"):
                self.model.generation_config.language = None
            if hasattr(self.model.generation_config, "task"):
                self.model.generation_config.task = None
            if hasattr(self.model.generation_config, "return_timestamps"):
                self.model.generation_config.return_timestamps = False

    def _configure_tuning_mode(self) -> None:
        if self.tuning_mode == "prefix":
            freeze_base_model_parameters(self.model)
            prefix_cfg = self.tuning_cfg.get("prefix", {})
            self.prefix_manager = WhisperPrefixTuningManager(
                encoder_layers=len(self.model.model.encoder.layers),
                decoder_layers=len(self.model.model.decoder.layers),
                encoder_heads=int(self.model.config.encoder_attention_heads),
                decoder_heads=int(self.model.config.decoder_attention_heads),
                d_model=int(self.model.config.d_model),
                encoder_prefix_length=int(prefix_cfg.get("encoder_prefix_length", 10)),
                decoder_prefix_length=int(prefix_cfg.get("decoder_prefix_length", 30)),
                init_std=float(prefix_cfg.get("init_std", 0.02)),
            )
            self.prefix_manager.attach(self.model)
            for parameter in self.prefix_manager.parameters():
                parameter.requires_grad = True
        elif self.tuning_mode == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
        elif self.tuning_mode == "lora":
            raise NotImplementedError("tuning.mode='lora' is reserved but not implemented in whisper_qa yet.")
        else:
            raise ValueError(f"Unsupported tuning mode: {self.tuning_mode}")

    def trainable_parameter_report(self) -> Dict[str, int]:
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = count_trainable_parameters(self.model)
        report = {
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        if self.prefix_manager is not None:
            report.update(self.prefix_manager.summary())
        return report

    def encode_audio(self, input_features: torch.Tensor) -> BaseModelOutput:
        return self.model.model.encoder(input_features=input_features, return_dict=True)

    def compute_asr_loss(
        self,
        encoder_outputs: BaseModelOutput,
        asr_labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            labels=asr_labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss

    def build_transcript_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        transcript_cache_ids: torch.Tensor,
    ) -> EncoderDecoderCache:
        if transcript_cache_ids.ndim == 1:
            transcript_cache_ids = transcript_cache_ids.unsqueeze(0)

        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        decoder_outputs = self.model.model.decoder(
            input_ids=transcript_cache_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return decoder_outputs.past_key_values

    def tokenize_prompt(self, prompt_text: str) -> torch.Tensor:
        token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def tokenize_answer(self, answer_text: str) -> torch.Tensor:
        token_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        eos_token_id = int(getattr(self.tokenizer, "eos_token_id"))
        return torch.tensor(token_ids + [eos_token_id], dtype=torch.long, device=self.device)

    def score_answer_from_cache(
        self,
        transcript_cache: EncoderDecoderCache,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if target_ids.ndim == 1:
            target_ids = target_ids.unsqueeze(0)

        if prompt_ids.shape[1] == 0:
            raise ValueError("Prompt ids must contain at least one token.")
        if target_ids.shape[1] == 0:
            raise ValueError("Target ids must contain at least one token.")

        prompt_len = prompt_ids.shape[1]
        decoder_input_ids = prompt_ids
        if target_ids.shape[1] > 1:
            decoder_input_ids = torch.cat([prompt_ids, target_ids[:, :-1]], dim=1)

        cache_copy = clone_encoder_decoder_cache(transcript_cache)
        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=None,
            past_key_values=cache_copy,
            use_cache=True,
            return_dict=True,
        )
        logits = self.model.proj_out(decoder_outputs.last_hidden_state)
        answer_logits = logits[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :]
        gathered = gather_target_log_probs(answer_logits, target_ids)
        return {
            "log_prob": gathered.sum(dim=-1),
            "loss": -gathered.mean(),
            "token_log_probs": gathered,
        }

    def score_questions(
        self,
        transcript_cache: EncoderDecoderCache,
        question_bank: Any,
    ) -> Dict[str, Any]:
        positive_ids = self.tokenize_answer(question_bank.positive_answer)
        negative_ids = self.tokenize_answer(question_bank.negative_answer)

        label_scores: Dict[str, float] = {}
        raw_scores: Dict[str, Any] = {}
        for label, questions in question_bank.all_questions().items():
            entries = []
            for question in questions:
                prompt_ids = self.tokenize_prompt(question_bank.format_prompt(question))
                positive_result = self.score_answer_from_cache(transcript_cache, prompt_ids, positive_ids)
                negative_result = self.score_answer_from_cache(transcript_cache, prompt_ids, negative_ids)
                score = float((positive_result["log_prob"] - negative_result["log_prob"]).item())
                entries.append(
                    {
                        "question": question,
                        "positive_log_prob": float(positive_result["log_prob"].item()),
                        "negative_log_prob": float(negative_result["log_prob"].item()),
                        "score": score,
                    }
                )
            raw_scores[label] = entries
            label_scores[label] = float(sum(entry["score"] for entry in entries) / max(len(entries), 1))
        return {
            "label_scores": label_scores,
            "raw_question_scores": raw_scores,
        }

    def _manual_greedy_generate(
        self,
        encoder_outputs: BaseModelOutput,
        decoder_input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        generated = decoder_input_ids
        eos_token_id = int(self.model.config.eos_token_id)
        for _ in range(max_new_tokens):
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=generated,
                use_cache=False,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool(torch.all(next_token.eq(eos_token_id))):
                break
        return generated

    def generate_transcript(self, input_features: torch.Tensor) -> tuple[str, BaseModelOutput]:
        encoder_outputs = self.encode_audio(input_features)
        forced_decoder_ids = list(getattr(self.model.generation_config, "forced_decoder_ids", None) or [])
        prompt_ids = [int(self.model.config.decoder_start_token_id)] + [int(token_id) for _, token_id in forced_decoder_ids]
        decoder_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        max_new_tokens = int(self.model_cfg.get("transcript_generation_max_new_tokens", 160))

        original_forced_decoder_ids = getattr(self.model.generation_config, "forced_decoder_ids", None)
        original_model_forced_decoder_ids = getattr(self.model.config, "forced_decoder_ids", None)
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None
        try:
            try:
                generated_ids = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            except Exception as exc:
                logger.warning("Falling back to manual greedy transcript decoding: %s", exc)
                generated_ids = self._manual_greedy_generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=max_new_tokens,
                )
        finally:
            self.model.generation_config.forced_decoder_ids = original_forced_decoder_ids
            self.model.config.forced_decoder_ids = original_model_forced_decoder_ids

        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return transcript, encoder_outputs

    def predict_single(self, input_features: torch.Tensor, question_bank: Any) -> Dict[str, Any]:
        total_start = time.perf_counter()
        asr_start = time.perf_counter()
        transcript, encoder_outputs = self.generate_transcript(input_features)
        asr_elapsed_ms = (time.perf_counter() - asr_start) * 1000.0

        transcript_cache_ids = encode_transcript_cache_ids(self.tokenizer, transcript, self.model_cfg)
        transcript_cache_tensor = torch.tensor(transcript_cache_ids, dtype=torch.long, device=self.device)

        qa_start = time.perf_counter()
        transcript_cache = self.build_transcript_cache(
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            transcript_cache_ids=transcript_cache_tensor,
        )
        scored = self.score_questions(transcript_cache=transcript_cache, question_bank=question_bank)
        qa_elapsed_ms = (time.perf_counter() - qa_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        predicted_label = max(scored["label_scores"], key=scored["label_scores"].get)
        return {
            "predicted_label": predicted_label,
            "label_scores": scored["label_scores"],
            "transcript": transcript,
            "transcript_source": str(self.inference_cfg.get("transcript_source", "whisper_asr_greedy")),
            "raw_question_scores": scored["raw_question_scores"],
            "latency_ms": {
                "asr_ms": asr_elapsed_ms,
                "qa_ms": qa_elapsed_ms,
                "total_ms": total_elapsed_ms,
            },
            "warnings": [],
        }

    def build_checkpoint_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tuning_mode": self.tuning_mode,
            "model_name": self.model_name,
        }
        if self.tuning_mode == "prefix":
            if self.prefix_manager is None:
                raise RuntimeError("Prefix mode requested, but prefix manager is not initialized.")
            payload["prefix_state_dict"] = self.prefix_manager.state_dict()
        elif self.tuning_mode == "full":
            payload["model_state_dict"] = self.model.state_dict()
        return payload

    def load_checkpoint_payload(self, payload: Mapping[str, Any]) -> None:
        if self.tuning_mode == "prefix":
            if self.prefix_manager is None:
                raise RuntimeError("Prefix mode requested, but prefix manager is not initialized.")
            self.prefix_manager.load_state_dict(payload["prefix_state_dict"])
        elif self.tuning_mode == "full":
            self.model.load_state_dict(payload["model_state_dict"])
