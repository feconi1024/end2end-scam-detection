"""
Entry point for the Speech Language Model scam detector.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
_speech_lm_root = Path(__file__).resolve().parent
if str(_speech_lm_root) not in sys.path:
    sys.path.insert(0, str(_speech_lm_root))

from src.pipeline import run_pipeline

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    Fore = type("Dummy", (), {"RED": "", "__getattr__": lambda *_: ""})()
    Style = type("Dummy", (), {"BRIGHT": "", "RESET_ALL": "", "__getattr__": lambda *_: ""})()
    _HAS_COLOR = False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SLM-based scam detection on raw telephony audio"
    )
    parser.add_argument(
        "--audio_file",
        "-a",
        type=Path,
        required=True,
        help="Path to the target audio file (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=Path,
        default=None,
        help="Path to system prompt (default: config/prompt.txt)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colorized scam warning",
    )
    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        return 1

    try:
        result = run_pipeline(
            audio_path=args.audio_file,
            config_path=args.config,
            prompt_path=args.prompt,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))

    is_scam = result.get("is_scam")
    if is_scam is True and not args.no_color and _HAS_COLOR:
        print(
            f"\n{Fore.RED}{Style.BRIGHT}⚠ SCAM DETECTED{Style.RESET_ALL}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
