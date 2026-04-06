from __future__ import annotations

import unittest
from pathlib import Path

from whisper_qa.src.questions import QuestionBank


class QuestionBankTests(unittest.TestCase):
    def test_loads_question_bank(self):
        bank_path = Path("whisper_qa") / "config" / "questions_zh.yaml"
        bank = QuestionBank.from_yaml(bank_path)
        self.assertIn("scam", bank.labels)
        self.assertIn("non_scam", bank.labels)
        self.assertGreaterEqual(len(bank.labels["scam"]), 2)

    def test_paraphrase_sampling_is_deterministic(self):
        bank_path = Path("whisper_qa") / "config" / "questions_zh.yaml"
        bank = QuestionBank.from_yaml(bank_path)
        first = bank.sample_training_questions(seed=123)
        second = bank.sample_training_questions(seed=123)
        third = bank.sample_training_questions(seed=124)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
