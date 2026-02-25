## Cascading Zero-Shot Scam Detection System

This project implements a **Zero-Shot Cascading Scam Detection System**:

- **ASR Layer**: `faster-whisper` for fast, multilingual transcription with VAD.
- **Reasoning Layer**: Either **Hugging Face Transformers** (local models) or an **OpenAI-compatible API** (local/cloud) for fraud analysis.
- **Pipeline**: Audio file → Transcript → JSON classification (`Scam` vs `Legitimate`).

### Structure

- `config/settings.yaml`: Model paths, LLM backend and endpoint, thresholds.
- `config/prompt.txt`: System prompt for scam analysis.
- `src/audio_processor.py`: Audio loading, resampling, mono conversion, -3 dB normalization.
- `src/asr_engine.py`: Faster-Whisper wrapper.
- `src/llm_engine.py`: OpenAI-compatible LLM client and `get_llm_engine()` factory.
- `src/llm_engine_hf.py`: Hugging Face Transformers LLM engine (local models).
- `src/config_loader.py`: Pydantic-backed settings loader.
- `src/pipeline.py`: End-to-end orchestration.
- `main.py`: CLI entry point.

### Installation

```bash
cd cascading
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### Configuration

**LLM backend** (in `config/settings.yaml`):

- `llm.backend: "huggingface"` — Use Hugging Face Transformers (e.g. `Qwen/Qwen2.5-0.5B-Instruct`). No API key; runs locally.
- `llm.backend: "openai"` — Use an OpenAI-compatible API (vLLM, OpenAI, etc.). Set `base_url` and `api_key`.

Edit `config/settings.yaml` or override via environment variables:

- `LLM_API_KEY` / `OPENAI_API_KEY` (OpenAI backend)
- `LLM_BASE_URL` (OpenAI backend)
- `LLM_MODEL` (model name or HF model id)

### Usage

```bash
python -m main --input_file path/to/call_audio.wav
```

Optional custom config:

```bash
python -m main --input_file path/to/call_audio.wav --config config/settings.yaml
```

