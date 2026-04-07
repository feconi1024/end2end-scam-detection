from __future__ import annotations

import unittest
from pathlib import Path

import torch

from whisper_qa.src.model import WhisperQAModel, gather_target_log_probs
from whisper_qa.tests.test_support import ToyProcessor, build_test_config, build_tiny_whisper_model


class PrefixAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.processor = ToyProcessor(feature_length=8)
        self.config = build_test_config(Path('.'))
        self.base_model = build_tiny_whisper_model()
        self.model = WhisperQAModel(
            processor=self.processor,
            config=self.config,
            whisper_model=self.base_model,
        )

    def test_prefix_attachment_coverage(self):
        encoder_attn = self.model.model.model.encoder.layers[0].self_attn
        decoder_attn = self.model.model.model.decoder.layers[0].self_attn
        self.assertTrue(getattr(encoder_attn, "_whisper_qa_prefix_patched", False))
        self.assertTrue(getattr(decoder_attn, "_whisper_qa_prefix_patched", False))

    def test_frozen_parameter_enforcement(self):
        for name, parameter in self.model.model.named_parameters():
            if "whisper_qa_prefix_manager" in name:
                self.assertTrue(parameter.requires_grad, msg=name)
            else:
                self.assertFalse(parameter.requires_grad, msg=name)

    def test_transcript_cache_reuse_without_encoder_outputs(self):
        input_features = torch.randn(1, 80, 8)
        encoder_outputs = self.model.encode_audio(input_features)
        transcript_cache = self.model.build_transcript_cache(
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            transcript_cache_ids=torch.tensor([1, 6, 7], dtype=torch.long),
        )
        result = self.model.score_answer_from_cache(
            transcript_cache=transcript_cache,
            prompt_ids=torch.tensor([8, 9], dtype=torch.long, device=self.model.device),
            target_ids=torch.tensor([10, 2], dtype=torch.long, device=self.model.device),
        )
        self.assertTrue(torch.isfinite(result["log_prob"]).all())
        self.assertEqual(tuple(result["token_log_probs"].shape), (1, 2))

    def test_yes_no_log_prob_gathering(self):
        logits = torch.log(torch.tensor([[[0.1, 0.9], [0.8, 0.2]]], dtype=torch.float32))
        target_ids = torch.tensor([[1, 0]], dtype=torch.long)
        gathered = gather_target_log_probs(logits, target_ids)
        expected = torch.log(torch.tensor([[0.9, 0.8]], dtype=torch.float32))
        self.assertTrue(torch.allclose(gathered, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
