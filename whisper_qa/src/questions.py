from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Mapping

import yaml


@dataclass(frozen=True)
class QuestionBank:
    language: str
    positive_answer: str
    negative_answer: str
    prompt_template: str
    labels: Dict[str, List[str]]

    @classmethod
    def from_yaml(cls, path: Path) -> "QuestionBank":
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        labels_raw = raw.get("labels", {})
        labels: Dict[str, List[str]] = {}
        for label, payload in labels_raw.items():
            questions = payload.get("questions", [])
            if not questions:
                raise ValueError(f"Label '{label}' in {path} must define at least one question.")
            labels[str(label)] = [str(question) for question in questions]

        return cls(
            language=str(raw.get("language", "zh")),
            positive_answer=str(raw.get("positive_answer", "是")),
            negative_answer=str(raw.get("negative_answer", "否")),
            prompt_template=str(raw.get("prompt_template", "问题：{question} 答案：")),
            labels=labels,
        )

    def format_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def sample_training_questions(self, seed: int) -> Dict[str, str]:
        rng = Random(seed)
        return {
            label: questions[rng.randrange(len(questions))]
            for label, questions in self.labels.items()
        }

    def all_questions(self) -> Mapping[str, List[str]]:
        return self.labels
