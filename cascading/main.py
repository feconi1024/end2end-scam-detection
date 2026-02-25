from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colorama import Fore, Style, init as colorama_init

from src.config_loader import load_settings
from src.pipeline import ScamDetectionPipeline, PipelineResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Shot Cascading Scam Detection System",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input audio file (wav/mp3/m4a).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML settings file.",
    )
    return parser.parse_args(argv)


def print_report(pipeline_result: PipelineResult) -> None:
    colorama_init(autoreset=True)

    result = pipeline_result.analysis

    is_unknown = str(getattr(result, "category", "")).lower() == "unknown"
    is_scam = bool(getattr(result, "is_scam", False))

    if is_unknown:
        color = Fore.YELLOW
        status = "UNKNOWN (AUDIO/TRANSCRIPT ISSUE)"
    else:
        color = Fore.RED if is_scam else Fore.GREEN
        status = "SCAM" if is_scam else "LEGITIMATE / SAFE"

    print(color + f"\n=== Call Classification: {status} ===" + Style.RESET_ALL)
    print(f"Category      : {result.category}")
    print(f"Risk Score    : {result.risk_score}")
    print(f"Urgency Flag  : {result.urgency_detected}")
    print(f"Reasoning     : {result.reasoning}")

    if result.flagged_phrases:
        print("Flagged Phrases:")
        for phrase in result.flagged_phrases:
            print(f"  - {phrase}")

    # Timing information
    audio_dur = pipeline_result.audio_duration_s
    asr_t = pipeline_result.asr_time_s
    llm_t = pipeline_result.llm_time_s
    total_t = pipeline_result.total_time_s

    print("\nTiming:")
    print(f"  Audio duration    : {audio_dur:.2f} s")
    print(f"  ASR time          : {asr_t:.2f} s")
    print(f"  LLM time          : {llm_t:.2f} s")
    print(f"  Total pipeline    : {total_t:.2f} s")

    if audio_dur > 0:
        minutes = audio_dur / 60.0
        print(f"  ASR time / min    : {asr_t / minutes:.2f} s / min audio")
        print(f"  LLM time / min    : {llm_t / minutes:.2f} s / min audio")
        print(f"  Total time / min  : {total_t / minutes:.2f} s / min audio")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_settings(args.config) if args.config else load_settings()
    pipeline = ScamDetectionPipeline(cfg)

    audio_path = Path(args.input_file)
    if not audio_path.exists():
        print(f"Input file does not exist: {audio_path}", file=sys.stderr)
        return 1

    try:
        result = pipeline.run(audio_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error while running scam detection pipeline: {exc}", file=sys.stderr)
        return 1

    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

