#!/usr/bin/env python3
"""
Extract transcription-friendly audio from a single video or a directory tree of videos.

Default output format is mono 16 kHz WAV, which is a good fit for ASR.

Examples:
    python extract_audio.py video.mp4
    python extract_audio.py /path/to/video_dir --output-root extracted_audio
    python extract_audio.py /path/to/video_dir --in-place
"""

import argparse
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def collect_video_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    raise ValueError(f"Input path does not exist: {input_path}")


def build_output_path(video_path: Path, input_root: Path, output_root: Path | None, in_place: bool, ext: str):
    if in_place:
        return video_path.with_suffix(ext)
    if output_root is None:
        output_root = Path("audio_output")
    if input_root.is_file():
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root / f"{video_path.stem}{ext}"
    rel = video_path.relative_to(input_root).with_suffix(ext)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000, channels: int = 1):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Extract mono 16 kHz WAV audio from video files")
    parser.add_argument("input", help="Video file or directory containing videos")
    parser.add_argument("--output-root", default=None, help="Root output directory when not using --in-place")
    parser.add_argument("--in-place", action="store_true", help="Write audio next to each source video")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Output channels (default: 1)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_root = Path(args.output_root).expanduser() if args.output_root else None

    try:
        videos = collect_video_files(input_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print(f"No video files found in {input_path}")
        sys.exit(0)

    print(f"Found {len(videos)} video file(s)")
    done = 0
    skipped = 0
    failed = 0

    for i, video_path in enumerate(videos, 1):
        output_path = build_output_path(video_path, input_path, output_root, args.in_place, ".wav")
        if args.skip_existing and output_path.exists():
            print(f"[{i}/{len(videos)}] Skip existing: {output_path}")
            skipped += 1
            continue
        try:
            extract_audio(video_path, output_path, sample_rate=args.sample_rate, channels=args.channels)
            print(f"[{i}/{len(videos)}] OK  {video_path} -> {output_path}")
            done += 1
        except Exception as e:
            print(f"[{i}/{len(videos)}] FAIL {video_path}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print("Audio extraction complete.")
    print(f"  Extracted : {done}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")


if __name__ == "__main__":
    main()
