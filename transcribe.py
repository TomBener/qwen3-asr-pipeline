#!/usr/bin/env python3
"""
Qwen3-ASR Transcription Pipeline

Transcribes audio files using Qwen3-ASR-1.7B on Apple Silicon (MPS).
Outputs structured results for academic use:
  - Plain text (.txt) — raw transcription for topic modeling / LLM prompts
  - JSON with metadata, segments, and word-level timestamps (.json)

To fit within Apple Silicon unified memory (16–24 GB), ASR and forced
alignment run in two sequential phases — the ASR model is fully unloaded
before the aligner is loaded.

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --language Chinese
    python transcribe.py audio.mp3 --output-dir results
    python transcribe.py audio.mp3 --no-timestamps
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


def get_device_and_dtype():
    """Select the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    else:
        return "cpu", torch.float32


def unload_model(model):
    """Aggressively free model memory."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def group_words_into_segments(time_stamps, max_gap: float = 0.5, max_duration: float = 8.0):
    """Group word-level timestamps into subtitle segments.

    Args:
        time_stamps: List of ForcedAlignItem with .text, .start_time, .end_time
        max_gap: Maximum silence gap (seconds) before starting a new segment.
        max_duration: Maximum segment duration (seconds).
    """
    if not time_stamps:
        return []

    segments = []
    current_words = [time_stamps[0]]
    seg_start = time_stamps[0].start_time

    for item in time_stamps[1:]:
        prev_end = current_words[-1].end_time
        gap = item.start_time - prev_end
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
    """Write plain-text transcription."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def write_json(text: str, language: str, metadata: dict, segments: list,
               word_timestamps: list, path: Path):
    """Write structured JSON output for downstream LLM processing.

    Structure:
      metadata  — provenance, model info, timing
      result    — language, full text, timed segments, word timestamps
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Qwen3-ASR-1.7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument("--language", default=None,
                        help="Force language (e.g. Chinese, English). Default: auto-detect")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--no-timestamps", action="store_true",
                        help="Skip forced alignment (faster, no word timestamps)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    # Device setup
    device, dtype = get_device_and_dtype()
    print(f"Device: {device} | dtype: {dtype}")

    # ── Phase 1: ASR transcription (Qwen3-ASR-1.7B) ────────────────
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

    print(f"Transcribing: {audio_path}")
    t1 = time.time()

    results = asr_model.transcribe(
        audio=str(audio_path),
        language=args.language,
    )
    transcribe_time = time.time() - t1

    text = results[0].text
    language = results[0].language
    print(f"Transcription completed in {transcribe_time:.1f}s")
    print(f"Detected language: {language}")

    # Free ASR model before loading aligner
    unload_model(asr_model)
    print("ASR model unloaded.")

    # ── Phase 2: Forced alignment (separate pass to avoid OOM) ──────
    word_timestamps = []
    segments = []
    align_time = 0.0

    if not args.no_timestamps:
        from qwen_asr import Qwen3ForcedAligner

        aligner_name = "Qwen/Qwen3-ForcedAligner-0.6B"
        print(f"\n[Phase 2] Loading forced aligner: {aligner_name}")
        t2 = time.time()

        aligner = Qwen3ForcedAligner.from_pretrained(
            aligner_name,
            dtype=dtype,
            device_map=device,
        )
        print(f"Aligner loaded in {time.time() - t2:.1f}s")

        print("Aligning word timestamps...")
        t3 = time.time()

        align_results = aligner.align(
            audio=str(audio_path),
            text=text,
            language=language,
        )
        align_time = time.time() - t3
        print(f"Alignment completed in {align_time:.1f}s")

        if align_results and align_results[0]:
            for item in align_results[0]:
                word_timestamps.append({
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                })
            segments = group_words_into_segments(list(align_results[0]))

        unload_model(aligner)
        print("Aligner unloaded.")

    # ── Write outputs ───────────────────────────────────────────────
    metadata = {
        "source_file": str(audio_path.resolve()),
        "asr_model": ASR_MODEL,
        "forced_aligner": None if args.no_timestamps else "Qwen/Qwen3-ForcedAligner-0.6B",
        "language_detected": language,
        "language_forced": args.language,
        "device": device,
        "dtype": str(dtype),
        "transcription_time_seconds": round(transcribe_time, 2),
        "alignment_time_seconds": round(align_time, 2),
        "timestamp": datetime.now().isoformat(),
    }

    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    write_txt(text, txt_path)
    write_json(text, language, metadata, segments, word_timestamps, json_path)
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")

    # Print transcription
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
