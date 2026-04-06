from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path

from torch.utils.data import DataLoader

from whisper_qa.src.data import (
    TeleAntiFraudManifestDataset,
    WhisperQACollator,
    load_manifest_records,
)
from whisper_qa.src.evaluation import run_evaluation
from whisper_qa.src.model import WhisperQAModel
from whisper_qa.src.questions import QuestionBank
from whisper_qa.src.trainer import WhisperQATrainer
from whisper_qa.tests.test_support import (
    ToyProcessor,
    build_test_config,
    build_tiny_whisper_model,
    write_tiny_wav,
)


class DataAndSmokeTests(unittest.TestCase):
    def _make_workspace_tmpdir(self) -> Path:
        root = Path("whisper_qa") / "tests" / "_tmp"
        root.mkdir(parents=True, exist_ok=True)
        path = root / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _create_manifest_fixture(self, root: Path) -> Path:
        manifest_root = root / "TeleAntiFraud-28k" / "corrected_manifests"
        audio_root = root / "TeleAntiFraud-28k" / "TeleAntiFraud-28k" / "merged_result"
        write_tiny_wav(audio_root / "a.wav")
        write_tiny_wav(audio_root / "b.wav")
        manifest_root.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "val"):
            manifest_path = manifest_root / f"{split_name}_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "transcript"])
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "TeleAntiFraud-28k/merged_result/a.wav",
                        "label": "scam",
                        "transcript": "请转账",
                    }
                )
                writer.writerow(
                    {
                        "path": "TeleAntiFraud-28k/merged_result/b.wav",
                        "label": "non_scam",
                        "transcript": "正常通知",
                    }
                )
        return manifest_root

    def test_nested_teleantifraud_path_resolution(self):
        tmp_path = self._make_workspace_tmpdir()
        try:
            manifest_root = self._create_manifest_fixture(tmp_path)
            records = load_manifest_records(manifest_root, split_name="train")
            self.assertTrue(Path(records[0].audio_path).exists())
            self.assertIn("TeleAntiFraud-28k", records[0].audio_path)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_smoke_training_inference_and_evaluation(self):
        tmp_path = self._make_workspace_tmpdir()
        try:
            manifest_root = self._create_manifest_fixture(tmp_path)
            config = build_test_config(tmp_path)
            config["data"]["dataset_path"] = str(manifest_root)
            config["training"]["output_dir"] = str(tmp_path / "outputs")
            config["evaluation"]["output_dir"] = str(tmp_path / "outputs" / "eval")

            processor = ToyProcessor(feature_length=8)
            question_bank = QuestionBank.from_yaml(Path("whisper_qa") / "config" / "questions_zh.yaml")
            collator = WhisperQACollator(processor=processor, config=config)

            train_records = load_manifest_records(manifest_root, split_name="train")
            val_records = load_manifest_records(manifest_root, split_name="val")
            train_loader = DataLoader(
                TeleAntiFraudManifestDataset(train_records),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collator,
            )
            val_loader = DataLoader(
                TeleAntiFraudManifestDataset(val_records),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collator,
            )

            model = WhisperQAModel(
                processor=processor,
                config=config,
                whisper_model=build_tiny_whisper_model(),
            )
            trainer = WhisperQATrainer(
                model=model,
                processor=processor,
                question_bank=question_bank,
                train_dataloader=train_loader,
                eval_dataloader=None,
                config=config,
                output_dir=Path(config["training"]["output_dir"]),
            )
            history = trainer.train()
            self.assertTrue(Path(history["last_checkpoint"]).exists())

            model.eval()
            batch = next(iter(val_loader))
            result = model.predict_single(batch["input_features"][0:1], question_bank=question_bank)
            self.assertIn(result["predicted_label"], {"scam", "non_scam"})
            self.assertIn("label_scores", result)

            eval_dir = tmp_path / "eval_artifacts"
            eval_result = run_evaluation(
                model=model,
                dataloader=val_loader,
                question_bank=question_bank,
                output_dir=eval_dir,
                split_name="val",
            )
            self.assertTrue((eval_dir / "metrics.json").exists())
            self.assertTrue((eval_dir / "predictions.jsonl").exists())
            self.assertTrue((eval_dir / "predictions.csv").exists())
            self.assertIn("macro_f1", eval_result["metrics"])
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
