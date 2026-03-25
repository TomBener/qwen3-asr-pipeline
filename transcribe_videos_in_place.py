#!/usr/bin/env python3
"""
Recursively transcribe video files and write outputs next to each video.

For each video file found under the input directory, this script:
1. extracts ASR-friendly audio to <video_stem>.wav beside the video
2. runs Qwen3-ASR models with periodic reloads to reduce MPS memory buildup
3. routes long audio to CPU to avoid MPS OOM / SIGABRT on huge buffers
4. writes <video_stem>.txt and <video_stem>.json beside the video
5. writes <video_stem>.error beside the video if ASR fails
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


def get_device_and_dtype(prefer_cpu: bool = False):
    if prefer_cpu:
        return "cpu", torch.float32
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


def unload_models(asr_model, aligner):
    if asr_model is not None:
        unload_model(asr_model)
    if aligner is not None:
        unload_model(aligner)


def load_models(device, dtype, no_timestamps: bool):
    print(f"\nLoading ASR model on {device} ({dtype})")
    t0 = time.time()
    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
    print(f"ASR model loaded in {time.time() - t0:.1f}s")

    aligner = None
    if not no_timestamps:
        from qwen_asr import Qwen3ForcedAligner

        print(f"Loading forced aligner on {device} ({dtype})")
        t1 = time.time()
        aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL,
            dtype=dtype,
            device_map=device,
        )
        print(f"Aligner loaded in {time.time() - t1:.1f}s")

    return asr_model, aligner


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


def write_error_log(video_path: Path, error: Exception):
    error_path = video_path.with_suffix(".error")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(f"file: {video_path}\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"error: {type(error).__name__}: {error}\n")


def is_done(video_path: Path) -> bool:
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("result", {}).get("text"))
    except Exception:
        return False


def has_errored(video_path: Path) -> bool:
    return video_path.with_suffix(".error").exists()


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
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def transcribe_with_models(asr_model, aligner, wav_path: Path, language_arg: str | None):
    t1 = time.time()
    results = asr_model.transcribe(audio=str(wav_path), language=language_arg)
    transcribe_time = time.time() - t1
    text = results[0].text
    language = results[0].language

    align_time = 0.0
    segments = []
    word_timestamps = []
    if aligner is not None:
        t2 = time.time()
        results = aligner.align(audio=str(wav_path), text=text, language=language)
        align_time = time.time() - t2
        if results and results[0]:
            for item in results[0]:
                word_timestamps.append({
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                })
            segments = group_words_into_segments(list(results[0]))

    return text, language, transcribe_time, align_time, segments, word_timestamps


def main():
    parser = argparse.ArgumentParser(description="Recursively transcribe videos and save outputs beside each video")
    parser.add_argument("input_dir", help="Directory containing videos")
    parser.add_argument("--language", default=None, help="Force language, e.g. Chinese")
    parser.add_argument("--no-timestamps", action="store_true", help="Skip forced alignment")
    parser.add_argument("--retry-errors", action="store_true", help="Retry files with existing .error logs")
    parser.add_argument("--reuse-existing-wav", action="store_true", help="Reuse <video_stem>.wav when it already exists")
    parser.add_argument("--reload-every", type=int, default=8, help="Reload models every N files to reduce MPS memory buildup")
    parser.add_argument("--cpu-threshold-seconds", type=float, default=240.0, help="Route audio longer than this to CPU")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    video_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        print(f"No video files found in {input_dir}")
        sys.exit(0)

    pending, skipped, errored = [], [], []
    for p in video_files:
        if is_done(p):
            skipped.append(p)
        elif has_errored(p) and not args.retry_errors:
            errored.append(p)
        else:
            pending.append(p)

    print(f"Found {len(video_files)} video file(s): {len(pending)} pending, {len(skipped)} already done, {len(errored)} previously failed")
    if not pending:
        print("Nothing to do.")
        sys.exit(0)

    device, dtype = get_device_and_dtype()
    print(f"Primary device: {device} | dtype: {dtype}")
    print(f"CPU fallback threshold: {args.cpu_threshold_seconds}s | Reload every: {args.reload_every} files")

    asr_model, aligner = load_models(device, dtype, args.no_timestamps)
    done_now = 0
    failed_now = 0
    processed_since_reload = 0

    for i, video_path in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] Processing: {video_path}")
        wav_path = video_path.with_suffix(".wav")
        txt_path = video_path.with_suffix(".txt")
        json_path = video_path.with_suffix(".json")
        err_path = video_path.with_suffix(".error")

        try:
            if wav_path.exists() and args.reuse_existing_wav:
                print(f"  Reusing WAV: {wav_path}")
            else:
                extract_audio(video_path, wav_path, overwrite=not args.reuse_existing_wav)
                print(f"  Audio ok | Saved: {wav_path}")

            duration = get_audio_duration_seconds(wav_path)
            print(f"  Audio duration: {duration:.1f}s")

            use_cpu = duration >= args.cpu_threshold_seconds
            run_device, run_dtype = get_device_and_dtype(prefer_cpu=use_cpu)

            if use_cpu:
                print(f"  Long audio -> CPU fallback")
                unload_models(asr_model, aligner)
                asr_model, aligner = load_models(run_device, run_dtype, args.no_timestamps)
                processed_since_reload = 0
            elif processed_since_reload >= args.reload_every:
                print("  Reloading models to reduce MPS memory buildup")
                unload_models(asr_model, aligner)
                asr_model, aligner = load_models(device, dtype, args.no_timestamps)
                processed_since_reload = 0

            text, language, transcribe_time, align_time, segments, word_timestamps = transcribe_with_models(
                asr_model, aligner, wav_path, args.language
            )
            print(f"  ASR ok | Language: {language} | Time: {transcribe_time:.1f}s | Preview: {text[:60]}{'...' if len(text) > 60 else ''}")
            if aligner is not None:
                print(f"  Align ok | Time: {align_time:.1f}s | Segments: {len(segments)}")

            metadata = {
                "source_file": str(video_path.resolve()),
                "asr_model": ASR_MODEL,
                "forced_aligner": None if args.no_timestamps else ALIGNER_MODEL,
                "language_detected": language,
                "language_forced": args.language,
                "device": run_device,
                "dtype": str(run_dtype),
                "transcription_time_seconds": round(transcribe_time, 2),
                "alignment_time_seconds": round(align_time, 2),
                "audio_duration_seconds": round(duration, 2),
                "timestamp": datetime.now().isoformat(),
            }
            write_txt(text, txt_path)
            write_json(text, language, metadata, segments, word_timestamps, json_path)
            if err_path.exists():
                err_path.unlink()
            print(f"  Saved: {txt_path}")
            print(f"  Saved: {json_path}")
            done_now += 1
            processed_since_reload += 1

            if use_cpu:
                print("  Switching models back to primary device")
                unload_models(asr_model, aligner)
                asr_model, aligner = load_models(device, dtype, args.no_timestamps)
                processed_since_reload = 0
            else:
                cleanup_memory()

        except Exception as e:
            print(f"  ERROR: {e}")
            write_error_log(video_path, e)
            failed_now += 1
            unload_models(asr_model, aligner)
            asr_model, aligner = load_models(device, dtype, args.no_timestamps)
            processed_since_reload = 0

    unload_models(asr_model, aligner)
    print("\nModels unloaded.")
    print("\n" + "=" * 50)
    print("Batch complete.")
    print(f"  Processed this run : {done_now}")
    print(f"  Failed this run    : {failed_now}")
    print(f"  Previously done    : {len(skipped)}")
    print(f"  Total done         : {len(skipped) + done_now} / {len(video_files)}")


if __name__ == "__main__":
    main()
