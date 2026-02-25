## End-to-end Scam Detection Experiments

This repository contains multiple systems and utilities for scam detection experiments over telephone audio, sharing the same external dataset but using different model architectures.

### Project layout

- **`cascading/`**: Cascaded pipeline (ASR → text LLM).
  - **`cascading/src/`**
    - `pipeline.py`: Defines `ScamDetectionPipeline` and `PipelineResult` for end-to-end inference.
    - `asr_engine.py`, `audio_processor.py`: Audio loading and ASR.
    - `llm_engine.py`, `llm_engine_hf.py`: Scam analysis using an LLM backend.
    - `config_loader.py`: Loads `cascading/config/settings.yaml` into a typed `AppConfig`.
  - **`cascading/config/`**
    - `settings.yaml`: Main configuration for ASR/LLM/audio.
    - `prompt.txt`: Prompt template for the LLM.
- **`speech_lm/`**: Single-stage Speech Language Model (SLM) scam detector.
  - **`speech_lm/src/`**
    - `pipeline.py`: `run_pipeline(...)` helper for end-to-end SLM inference.
    - `slm_engine.py`, `audio_utils.py`, `schemas.py`: Core SLM logic and audio utilities.
    - `config_loader.py`: Loads `speech_lm/config/settings.yaml` into a typed `SLMConfig`.
  - **`speech_lm/config/`**
    - `settings.yaml`: SLM model/generation/audio configuration.
    - `prompt.txt`: System prompt used for SLM inference.
- **`TeleAntiFraud-28k/`** (ignored by Git):
  - External dataset folder. Large audio data should live here or in a similar directory **outside version control**.
  - `sample_100_balanced.py`: CLI utility to sample a balanced 100-file subset from the dataset.
- **`sample_100_balanced/`** (ignored by Git):
  - Example output directory containing a small balanced subset of the TeleAntiFraud dataset for quick experiments.
- **`.gitignore`**:
  - Excludes large dataset directories (`TeleAntiFraud-28k/`) and sampled outputs (`sample_100_balanced/`) from the repository.

This layout keeps **code for different systems clearly separated** (`cascading` vs `speech_lm`), while allowing shared use of the same dataset via paths and configuration rather than hard-coding machine-specific locations.

### Using the modules from other code

You can import and call the pipelines directly from Python, without changing into each subfolder:

```python
from pathlib import Path

# Cascaded ASR + LLM system
from cascading.src.config_loader import load_settings as load_cascading_settings
from cascading.src.pipeline import ScamDetectionPipeline

cfg = load_cascading_settings()  # or pass an explicit config path
pipeline = ScamDetectionPipeline(config=cfg)
result = pipeline.run(Path("path/to/audio.mp3"))

print(result.analysis.is_scam, result.analysis.risk_score)

# Speech Language Model (SLM) system
from speech_lm.src.pipeline import run_pipeline as run_slm_pipeline

slm_result = run_slm_pipeline(audio_path="path/to/audio.mp3")
print(slm_result)
```

As long as this repository root is on `PYTHONPATH` (for example, by running from this directory or installing the project in editable mode), both `cascading` and `speech_lm` can be imported as normal Python packages.

### Dataset access and portability

The TeleAntiFraud-28k dataset is **not stored in Git**, only referenced by path:

- The sampling utility `TeleAntiFraud-28k/sample_100_balanced.py` accepts `--dataset-root` and `--output-dir` so you can point it at any dataset location on each machine.
- Large audio folders (the raw dataset and sampled subsets) are kept out of the repository via `.gitignore`, which makes the codebase **lightweight to clone** and easier to sync between local and remote servers.

Recommended pattern for portability:

- **Step 1**: Clone this repository on each machine (local laptop, remote GPU server, etc.).
- **Step 2**: Place or mount the TeleAntiFraud-28k dataset at a convenient path on that machine.
- **Step 3**: Use the CLI flag `--dataset-root` when running `sample_100_balanced.py` (or your own scripts) to point to the local dataset path, instead of hard-coding absolute paths in code.

This way, the **code remains identical across machines** (Git-managed), and only the dataset location differs, controlled by CLI flags or environment variables.

### GitHub / remote server workflow

- **Version-control only code and small config files**:
  - `cascading/`, `speech_lm/`, `*.py`, `*.yaml`, `*.txt`.
- **Keep datasets and generated audio out of Git**:
  - Already enforced via `.gitignore` (`TeleAntiFraud-28k/`, `sample_100_balanced/`).
- **Sync workflow**:
  - Commit and push from your local machine.
  - Pull on the remote server.
  - Ensure the dataset is present on the remote at some path, and pass that path into scripts via CLI or configuration.

With this structure and conventions, it is straightforward to add new modules or systems (e.g. another model family) alongside `cascading` and `speech_lm` while still sharing the same dataset and configuration patterns.

