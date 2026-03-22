# Qwen3-ASR Transcription Pipeline

A local transcription pipeline using [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) optimized for Apple Silicon (M3/M4). Built for academic research where transcription accuracy is the priority.

## Purpose

Transcribe audio files into structured text for downstream NLP tasks:

- **Named Entity Recognition (NER)** — extract people, locations, times, organizations
- **Topic modeling** — identify themes and subjects across a corpus
- **Text analysis** — sentiment, discourse analysis, keyword extraction
- **LLM-based information extraction** — feed structured transcripts to LLMs for further processing

## Output Formats

For an input file `audio.mp3`, the pipeline produces:

| File | Format | Use Case |
|------|--------|----------|
| `output/audio.txt` | Plain text | Direct input for topic modeling, text classification, LLM prompts |
| `output/audio.json` | JSON | Structured data with metadata, timed segments, and word-level timestamps for programmatic processing |

### JSON structure

The JSON output is designed for downstream LLM pipelines. `segments` provide natural time-bounded chunks suitable for per-segment NER or classification. `word_timestamps` provide fine-grained alignment for annotation tasks.

```json
{
  "metadata": {
    "source_file": "/absolute/path/to/audio.mp3",
    "asr_model": "Qwen/Qwen3-ASR-1.7B",
    "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "language_detected": "Chinese",
    "language_forced": null,
    "device": "mps",
    "dtype": "torch.bfloat16",
    "transcription_time_seconds": 4.4,
    "alignment_time_seconds": 0.6,
    "timestamp": "2026-03-22T15:30:00.000000"
  },
  "result": {
    "language": "Chinese",
    "text": "Full transcription text...",
    "segments": [
      {"start": 0.0, "end": 3.456, "text": "Segment text..."}
    ],
    "word_timestamps": [
      {"text": "嗯", "start_time": 0.12, "end_time": 0.34}
    ]
  }
}
```

## Requirements

- macOS with Apple Silicon (M3/M4)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv venv --python 3.12
uv pip install qwen-asr
source .venv/bin/activate
```

Models (~3.5 GB total) download automatically on first run and are cached in `~/.cache/huggingface/hub/`.

## Usage

### Single file

```bash
# Basic — auto-detect language, outputs to ./output/
python transcribe.py audio.mp3

# Force language
python transcribe.py audio.mp3 --language Chinese

# Custom output directory
python transcribe.py audio.mp3 --output-dir results

# Skip forced alignment (faster, no word timestamps)
python transcribe.py audio.mp3 --no-timestamps
```

### Batch (production)

```bash
# Transcribe all audio files in a directory — models loaded once
python batch_transcribe.py /path/to/audio/dir

# With options
python batch_transcribe.py /path/to/audio/dir --output-dir results --language Chinese

# Resume is automatic — already-transcribed files are skipped
python batch_transcribe.py /path/to/audio/dir   # re-run safely after a crash

# Retry previously failed files
python batch_transcribe.py /path/to/audio/dir --retry-errors
```

Supported formats: `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.ogg`

## Architecture

The pipeline runs in two sequential phases to fit within Apple Silicon unified memory (16–24 GB):

1. **Phase 1 — ASR** (`Qwen3-ASR-1.7B`): Transcribes audio to text. Model is then fully unloaded.
2. **Phase 2 — Forced Alignment** (`Qwen3-ForcedAligner-0.6B`): Aligns the transcribed text back to audio for word-level timestamps and segments.

This two-phase approach avoids out-of-memory errors that occur when both models are loaded simultaneously.

## Performance

Benchmarked with `bfloat16` on Apple M3 (24 GB). Expected similar or better on M4 (16 GB).

| Phase | Time | Memory |
|-------|------|--------|
| ASR (1.7B) | ~4–10s per minute of audio | ~5 GB |
| Forced alignment | ~1–2s per minute of audio | ~3 GB |
| **Peak** | | **~5 GB** |

- MPS (Metal Performance Shaders) backend is used automatically on Apple Silicon
- `bfloat16` precision — benchmarked 2–3.5x faster than `float32` on MPS with no accuracy difference
- For long audio (>10 min), increase `max_new_tokens` in `transcribe.py` if output is truncated
- In batch mode, models are loaded once and reused across all files — eliminates the ~10s per-file reload overhead

## Models

| Model | Parameters | Role |
|-------|-----------|------|
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | Primary ASR — best accuracy, 30 languages, 22 Chinese dialects |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | 0.6B | Word-level timestamp alignment |

## Downstream Usage Example

Feed the JSON output to an LLM for information extraction:

```python
import json

with open("output/audio.json") as f:
    data = json.load(f)

# Use full text for topic modeling or summarization
full_text = data["result"]["text"]

# Use segments for per-chunk NER or classification
for seg in data["result"]["segments"]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
```

## License

This pipeline script is provided as-is for research use. The Qwen3-ASR models are subject to their own [license terms](https://github.com/QwenLM/Qwen3-ASR).
