#!/usr/bin/env python3
"""
Transcribe a single video (or an existing sibling WAV) and write outputs beside it.

This script is meant to be run as an isolated subprocess by a parent orchestrator,
so each file gets a fresh Python process and fresh model state.
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def get_device_and_dtype(force_device: str | None = None):
    if force_device == "cpu":
        return "cpu", torch.float32
    if force_device == "mps":
        return "mps", torch.bfloat16
    if force_device == "cuda":
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def cleanup_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model(model):
    del model
    cleanup_memory()


def group_words_into_segments(time_stamps, max_gap: float = 0.5, max_duration: float = 8.0):
    if not time_stamps:
        return []
    segments = []
    current_words = [time_stamps[0]]
    seg_start = time_stamps[0].start_time
    for item in time_stamps[1:]:
        gap = item.start_time - current_words[-1].end_time
        duration = item.end_time - seg_start
        if gap > max_gap or duration > max_duration:
            segments.append({
                "start": round(seg_start, 3),
                "end": round(current_words[-1].end_time, 3),
                "text": "".join(w.text for w in current_words),
            })
            current_words = [item]
            seg_start = item.start_time
        else:
            current_words.append(item)
    if current_words:
        segments.append({
            "start": round(seg_start, 3),
            "end": round(current_words[-1].end_time, 3),
            "text": "".join(w.text for w in current_words),
        })
    return segments


def write_txt(text: str, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def write_json(text: str, language: str, metadata: dict, segments: list, word_timestamps: list, path: Path):
    output = {
        "metadata": metadata,
        "result": {
            "language": language,
            "text": text,
            "segments": segments,
            "word_timestamps": word_timestamps,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def write_error_log(video_path: Path, error_text: str):
    error_path = video_path.with_suffix(".error")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(f"file: {video_path}\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"error: {error_text}\n")


def extract_audio(video_path: Path, wav_path: Path, overwrite: bool = False):
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_audio_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Transcribe one video/audio target and write outputs beside it")
    parser.add_argument("video", help="Path to source video file")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--reuse-existing-wav", action="store_true")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"Error: unsupported video format: {video_path.suffix}", file=sys.stderr)
        sys.exit(1)

    wav_path = video_path.with_suffix(".wav")
    txt_path = video_path.with_suffix(".txt")
    json_path = video_path.with_suffix(".json")
    err_path = video_path.with_suffix(".error")

    try:
        if wav_path.exists() and args.reuse_existing_wav:
            print(f"Reusing WAV: {wav_path}")
        else:
            extract_audio(video_path, wav_path, overwrite=not args.reuse_existing_wav)
            print(f"Audio ok | Saved: {wav_path}")

        duration = get_audio_duration_seconds(wav_path)
        print(f"Audio duration: {duration:.1f}s")

        force_device = None if args.device == "auto" else args.device
        device, dtype = get_device_and_dtype(force_device)
        print(f"Device: {device} | dtype: {dtype}")

        t0 = time.time()
        asr_model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        print(f"ASR model loaded in {time.time() - t0:.1f}s")

        t1 = time.time()
        results = asr_model.transcribe(audio=str(wav_path), language=args.language)
        transcribe_time = time.time() - t1
        text = results[0].text
        language = results[0].language
        print(f"ASR ok | Language: {language} | Time: {transcribe_time:.1f}s")
        unload_model(asr_model)

        align_time = 0.0
        segments = []
        word_timestamps = []
        if not args.no_timestamps:
            from qwen_asr import Qwen3ForcedAligner
            t2 = time.time()
            aligner = Qwen3ForcedAligner.from_pretrained(
                ALIGNER_MODEL,
                dtype=dtype,
                device_map=device,
            )
            print(f"Aligner loaded in {time.time() - t2:.1f}s")
            t3 = time.time()
            align_results = aligner.align(audio=str(wav_path), text=text, language=language)
            align_time = time.time() - t3
            if align_results and align_results[0]:
                for item in align_results[0]:
                    word_timestamps.append({
                        "text": item.text,
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                    })
                segments = group_words_into_segments(list(align_results[0]))
            print(f"Align ok | Time: {align_time:.1f}s | Segments: {len(segments)}")
            unload_model(aligner)

        metadata = {
            "source_file": str(video_path.resolve()),
            "audio_file": str(wav_path.resolve()),
            "asr_model": ASR_MODEL,
            "forced_aligner": None if args.no_timestamps else ALIGNER_MODEL,
            "language_detected": language,
            "language_forced": args.language,
            "device": device,
            "dtype": str(dtype),
            "audio_duration_seconds": round(duration, 2),
            "transcription_time_seconds": round(transcribe_time, 2),
            "alignment_time_seconds": round(align_time, 2),
            "timestamp": datetime.now().isoformat(),
        }
        write_txt(text, txt_path)
        write_json(text, language, metadata, segments, word_timestamps, json_path)
        if err_path.exists():
            err_path.unlink()
        print(f"Saved: {txt_path}")
        print(f"Saved: {json_path}")
    except Exception as e:
        write_error_log(video_path, f"{type(e).__name__}: {e}")
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
