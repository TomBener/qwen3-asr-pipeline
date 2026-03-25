#!/usr/bin/env python3
"""
Two-step video transcription pipeline:
1. extract ASR-friendly WAV audio from videos
2. transcribe the extracted audio with the existing batch pipeline

Outputs are written next to each extracted WAV file inside the audio root.

Examples:
    python transcribe_videos.py ~/Downloads/douyin-favorites-batch-2026-03-24
    python transcribe_videos.py ~/Downloads/douyin-favorites-batch-2026-03-24 --audio-root ./extracted_audio --language Chinese
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Extract audio from videos, then batch transcribe")
    parser.add_argument("input_dir", help="Directory containing videos")
    parser.add_argument("--audio-root", default="audio_output", help="Directory to store extracted WAV files")
    parser.add_argument("--language", default=None, help="Force language, e.g. Chinese")
    parser.add_argument("--no-timestamps", action="store_true", help="Skip forced alignment")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed audio files")
    parser.add_argument("--skip-existing-audio", action="store_true", help="Do not re-extract WAV files that already exist")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    audio_root = Path(args.audio_root).expanduser()

    extract_cmd = [
        sys.executable,
        "extract_audio.py",
        str(input_dir),
        "--output-root",
        str(audio_root),
    ]
    if args.skip_existing_audio:
        extract_cmd.append("--skip-existing")

    transcribe_cmd = [
        sys.executable,
        "batch_transcribe.py",
        str(audio_root),
        "--output-dir",
        str(audio_root),
    ]
    if args.language:
        transcribe_cmd.extend(["--language", args.language])
    if args.no_timestamps:
        transcribe_cmd.append("--no-timestamps")
    if args.retry_errors:
        transcribe_cmd.append("--retry-errors")

    run(extract_cmd)
    run(transcribe_cmd)


if __name__ == "__main__":
    main()
