#!/usr/bin/env python3
"""
Stable long-run video transcription orchestrator.

Recommended for lower-memory Macs and long-running Douyin batches:
- one Python subprocess per video
- optional MPS-first, CPU-fallback retry for each file
- safe resume using existing .json outputs
- works with pre-extracted sibling WAV files
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def is_done(video_path: Path) -> bool:
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            isinstance(data, dict)
            and isinstance(data.get("metadata"), dict)
            and isinstance(data.get("result"), dict)
            and "text" in data["result"]
            and not has_errored(video_path)
        )
    except Exception:
        return False


def has_errored(video_path: Path) -> bool:
    return video_path.with_suffix(".error").exists()


def collect_videos(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def run_once(python_exe: str, video_path: Path, language: str | None, no_timestamps: bool, reuse_existing_wav: bool, device: str):
    cmd = [python_exe, "transcribe_one_in_place.py", str(video_path), "--device", device]
    if language:
        cmd.extend(["--language", language])
    if no_timestamps:
        cmd.append("--no-timestamps")
    if reuse_existing_wav:
        cmd.append("--reuse-existing-wav")
    return subprocess.run(cmd, capture_output=True, text=True)


def append_jsonl(path: Path | None, payload: dict):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Transcribe videos using one subprocess per file")
    parser.add_argument("input_dir", help="Directory containing videos")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--reuse-existing-wav", action="store_true")
    parser.add_argument("--cpu-threshold-seconds", type=float, default=180.0, help="Use CPU directly for audio >= threshold")
    parser.add_argument("--mps-first", action="store_true", help="Try MPS first for shorter files, then CPU on failure")
    parser.add_argument("--state-file", default=None, help="Optional JSONL progress log for resumable runs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    videos = collect_videos(input_dir)
    pending, skipped, errored = [], [], []
    for p in videos:
        if is_done(p):
            skipped.append(p)
        elif has_errored(p) and not args.retry_errors:
            errored.append(p)
        else:
            pending.append(p)

    print(f"Found {len(videos)} video file(s): {len(pending)} pending, {len(skipped)} already done, {len(errored)} previously failed")
    if not pending:
        print("Nothing to do.")
        return

    python_exe = sys.executable
    state_file = Path(args.state_file).expanduser() if args.state_file else None
    done_now = 0
    failed_now = 0

    for i, video_path in enumerate(pending, 1):
        wav_path = video_path.with_suffix('.wav')
        duration = None
        if wav_path.exists():
            probe = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(wav_path)
            ], capture_output=True, text=True)
            if probe.returncode == 0:
                try:
                    duration = float(probe.stdout.strip())
                except ValueError:
                    duration = None

        if duration is not None and duration >= args.cpu_threshold_seconds:
            primary_device = 'cpu'
            fallback_device = None
        elif args.mps_first:
            primary_device = 'mps'
            fallback_device = 'cpu'
        else:
            primary_device = 'cpu'
            fallback_device = None

        print(f"\n[{i}/{len(pending)}] {video_path}")
        if duration is not None:
            print(f"  duration={duration:.1f}s primary={primary_device} fallback={fallback_device}")
        else:
            print(f"  primary={primary_device} fallback={fallback_device}")

        attempts = []

        result = run_once(python_exe, video_path, args.language, args.no_timestamps, args.reuse_existing_wav, primary_device)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        attempts.append({
            "device": primary_device,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        })

        success = result.returncode == 0 and is_done(video_path)
        final_device = primary_device
        if not success and fallback_device:
            print(f"  retrying with {fallback_device}...")
            result = run_once(python_exe, video_path, args.language, args.no_timestamps, args.reuse_existing_wav, fallback_device)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            attempts.append({
                "device": fallback_device,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            })
            success = result.returncode == 0 and is_done(video_path)
            final_device = fallback_device

        if success:
            done_now += 1
        else:
            failed_now += 1

        append_jsonl(state_file, {
            "timestamp": datetime.now().isoformat(),
            "video": str(video_path),
            "wav": str(video_path.with_suffix('.wav')),
            "status": "done" if success else "failed",
            "duration_seconds": duration,
            "primary_device": primary_device,
            "final_device": final_device,
            "fallback_device": fallback_device,
            "attempts": attempts,
        })

    print("\n" + "=" * 50)
    print("Batch complete.")
    print(f"  Processed this run : {done_now}")
    print(f"  Failed this run    : {failed_now}")
    print(f"  Previously done    : {len(skipped)}")
    print(f"  Total done         : {len(skipped) + done_now} / {len(videos)}")


if __name__ == "__main__":
    main()
