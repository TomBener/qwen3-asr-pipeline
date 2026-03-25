#!/usr/bin/env python3
"""
Qwen3-ASR Batch Transcription Pipeline

Processes a directory of audio files using Qwen3-ASR-1.7B. Models are loaded
once and reused across all files. Supports resume: already-transcribed files
are skipped based on the presence of their output JSON.

Usage:
    python batch_transcribe.py /path/to/audio/dir
    python batch_transcribe.py /path/to/audio/dir --output-dir results
    python batch_transcribe.py /path/to/audio/dir --language Chinese
    python batch_transcribe.py /path/to/audio/dir --no-timestamps
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}


def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    else:
        return "cpu", torch.float32


def unload_model(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def write_json(text: str, language: str, metadata: dict, segments: list,
               word_timestamps: list, path: Path):
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


def build_output_paths(audio_path: Path, input_dir: Path, output_dir: Path):
    rel = audio_path.relative_to(input_dir)
    base = (output_dir / rel).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".txt"), base.with_suffix(".json"), base.with_suffix(".error")


def write_error_log(audio_path: Path, error: Exception, input_dir: Path, output_dir: Path):
    """Write a mirrored .error file so failed files are visible but not re-attempted."""
    _, _, error_path = build_output_paths(audio_path, input_dir, output_dir)
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(f"file: {audio_path}\n")
        f.write(f"time: {datetime.now().isoformat()}\n")
        f.write(f"error: {type(error).__name__}: {error}\n")


def is_done(audio_path: Path, input_dir: Path, output_dir: Path) -> bool:
    """A file is done if its mirrored JSON output exists and is valid."""
    _, json_path, _ = build_output_paths(audio_path, input_dir, output_dir)
    if not json_path.exists():
        return False
    try:
        with open(json_path) as f:
            data = json.load(f)
        return bool(data.get("result", {}).get("text"))
    except Exception:
        return False


def has_errored(audio_path: Path, input_dir: Path, output_dir: Path) -> bool:
    _, _, error_path = build_output_paths(audio_path, input_dir, output_dir)
    return error_path.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using Qwen3-ASR-1.7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--language", default=None,
                        help="Force language for all files (e.g. Chinese). Default: auto-detect")
    parser.add_argument("--no-timestamps", action="store_true",
                        help="Skip forced alignment (faster, no word timestamps)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry files that previously failed (.error files)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    audio_files = sorted(
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        sys.exit(0)

    # Partition into pending / skipped / errored
    pending, skipped, errored = [], [], []
    for f in audio_files:
        if is_done(f, input_dir, output_dir):
            skipped.append(f)
        elif has_errored(f, input_dir, output_dir) and not args.retry_errors:
            errored.append(f)
        else:
            pending.append(f)

    total = len(audio_files)
    print(f"\nFound {total} audio file(s): "
          f"{len(pending)} pending, {len(skipped)} already done, "
          f"{len(errored)} previously failed (use --retry-errors to retry)")

    if not pending:
        print("Nothing to do.")
        sys.exit(0)

    device, dtype = get_device_and_dtype()
    print(f"Device: {device} | dtype: {dtype}")

    # ── Phase 1: ASR — load once, transcribe all ────────────────────
    print(f"\n[Phase 1] Loading ASR model: {ASR_MODEL}")
    t0 = time.time()
    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=8,
        max_new_tokens=512,
    )
    print(f"ASR model loaded in {time.time() - t0:.1f}s")

    asr_results = {}  # audio_key -> (audio_path, text, language, transcribe_time)
    for i, audio_path in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] Transcribing: {audio_path.name}")
        t1 = time.time()
        try:
            results = asr_model.transcribe(
                audio=str(audio_path),
                language=args.language,
            )
            elapsed = time.time() - t1
            text = results[0].text
            language = results[0].language
            asr_results[str(audio_path.relative_to(input_dir))] = (audio_path, text, language, elapsed)
            print(f"  Language: {language} | Time: {elapsed:.1f}s | "
                  f"Preview: {text[:60]}{'...' if len(text) > 60 else ''}")
        except Exception as e:
            print(f"  ERROR during ASR: {e}")
            write_error_log(audio_path, e, input_dir, output_dir)

    unload_model(asr_model)
    print(f"\nASR complete. {len(asr_results)}/{len(pending)} succeeded. Model unloaded.")

    if not asr_results:
        print("No files to align. Exiting.")
        sys.exit(0)

    # ── Phase 2: Forced alignment — load once, align all ───────────
    align_results = {}  # stem -> (segments, word_timestamps, align_time)

    if not args.no_timestamps:
        from qwen_asr import Qwen3ForcedAligner

        print(f"\n[Phase 2] Loading forced aligner: {ALIGNER_MODEL}")
        t2 = time.time()
        aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL,
            dtype=dtype,
            device_map=device,
        )
        print(f"Aligner loaded in {time.time() - t2:.1f}s")

        for i, (audio_key, (audio_path, text, language, _)) in enumerate(asr_results.items(), 1):
            print(f"\n[{i}/{len(asr_results)}] Aligning: {audio_path.name}")
            t3 = time.time()
            try:
                results = aligner.align(
                    audio=str(audio_path),
                    text=text,
                    language=language,
                )
                elapsed = time.time() - t3
                word_timestamps = []
                segments = []
                if results and results[0]:
                    for item in results[0]:
                        word_timestamps.append({
                            "text": item.text,
                            "start_time": item.start_time,
                            "end_time": item.end_time,
                        })
                    segments = group_words_into_segments(list(results[0]))
                align_results[audio_key] = (segments, word_timestamps, elapsed)
                print(f"  Time: {elapsed:.1f}s | Segments: {len(segments)}")
            except Exception as e:
                print(f"  ERROR during alignment: {e}")
                # Don't write error log — ASR succeeded; just skip timestamps
                align_results[audio_key] = ([], [], 0.0)

        unload_model(aligner)
        print("\nAlignment complete. Aligner unloaded.")

    # ── Write outputs ───────────────────────────────────────────────
    print("\nWriting outputs...")
    for audio_key, (audio_path, text, language, transcribe_time) in asr_results.items():
        segments, word_timestamps, align_time = align_results.get(audio_key, ([], [], 0.0))

        metadata = {
            "source_file": str(audio_path.resolve()),
            "asr_model": ASR_MODEL,
            "forced_aligner": None if args.no_timestamps else ALIGNER_MODEL,
            "language_detected": language,
            "language_forced": args.language,
            "device": device,
            "dtype": str(dtype),
            "transcription_time_seconds": round(transcribe_time, 2),
            "alignment_time_seconds": round(align_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        txt_path, json_path, error_path = build_output_paths(audio_path, input_dir, output_dir)
        write_txt(text, txt_path)
        write_json(text, language, metadata, segments, word_timestamps, json_path)
        if error_path.exists():
            error_path.unlink()
        print(f"  Saved: {json_path}")

    # ── Summary ─────────────────────────────────────────────────────
    done_now = len(asr_results)
    failed_now = len(pending) - done_now
    print(f"\n{'='*50}")
    print(f"Batch complete.")
    print(f"  Processed this run : {done_now}")
    print(f"  Failed this run    : {failed_now}")
    print(f"  Previously done    : {len(skipped)}")
    print(f"  Total done         : {len(skipped) + done_now} / {total}")
    if failed_now:
        print(f"\nFailed files logged as .error in {output_dir}/")
        print("Re-run with --retry-errors to attempt them again.")


if __name__ == "__main__":
    main()
