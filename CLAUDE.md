# CLAUDE.md

Guidance for working in this repository.

## What this is

A command-line tool that turns a messy voiceover recording into a clean read that follows a written
script. The recording has pauses, false starts, and retakes; the tool keeps the **last good take**
of each sentence and stitches the pieces together with natural pauses. It always lets the user
review picks (`--dry-run`) before rendering audio.

## Pipeline

```
recording (audio/video) ──ffmpeg──► 16kHz mono WAV
                                          │
                                  faster-whisper (word-level timestamps, cached)
                                          │
script.txt ──► sentences ──► align each sentence to its last retake
                                          │
                          cut + concat with pauses ──► clean.wav
```

## Module map (`cleaner/`)

| File | Responsibility |
| --- | --- |
| `text.py` | `normalize()` (shared by script + transcript), `split_script()` into paragraph-aware sentences. Sentence splitting guards decimals/abbreviations. |
| `audio.py` | `extract_wav()` — ffmpeg to 16kHz mono temp WAV, with progress. `ensure_ffmpeg()`, `DependencyError`. |
| `transcribe.py` | faster-whisper word timestamps; JSON cache keyed on media+model+language; `resolve_device()` for auto/cpu/cuda + compute-type defaults. |
| `align.py` | **Core algorithm.** Forward-scanning, last-retake selection via take clustering. See below. |
| `report.py` | Dry-run table + MISSING / low-confidence flags + fix hints. |
| `assemble.py` | pydub cut/pad/pause/fade and export (format from extension). |
| `cli.py` | argparse, lazy heavy-imports (so `--help` works with nothing installed), orchestration, temp cleanup. |
| `util.py` | `fmt_ts()`, `Progress` (single-line progress for long stages). |

`clean.py` is the thin entry point (`python clean.py ...` / `uv run clean.py ...`).

## The alignment core (`align.py`)

- Maintain a pointer into the transcript word list; it only advances past a matched span.
- For each sentence, scan start positions forward, scoring word windows (lengths `len-2 .. len+max_extra`)
  against the normalized sentence with `rapidfuzz.fuzz.ratio`.
- Qualifying windows are grouped into **takes** (consecutive start positions, `_CLUSTER_GAP`). The
  **latest take** is chosen (that's the final retake), then the best-scoring window within it (so the
  first word isn't shaved off). This refines the spec's raw "latest start wins" rule.
- If nothing clears `--threshold`, retry at a relaxed threshold → `LOW`; still nothing → `MISSING`
  (never crash).

## Commands

Environment is managed with **uv**.

```bash
uv sync                                   # create .venv + install deps
uv run clean.py MEDIA SCRIPT --dry-run    # review picks, write no audio
uv run clean.py MEDIA SCRIPT -o clean.wav # render
uv sync --extra gpu                       # add CUDA cuBLAS/cuDNN for GPU (see README)
```

The transcript caches to `MEDIA.<model>.words.json`, so re-running the dry-run while tuning
`--threshold` skips re-transcription. `--refresh` forces a re-transcribe. Default `--device auto`
uses a GPU when present, else CPU (small.en on CPU ≈ real-time).

## Conventions

- **Atomic commits**; commit messages and code comments read as developer-written (no AI attribution
  trailers).
- **Long stages show progress** via `util.Progress` — never leave the terminal frozen.
- Heavy deps (faster-whisper, pydub) and ffmpeg are touched only when needed; `--help` and arg errors
  work without them, and missing pieces raise clear install hints.
- `data/` (sample recordings + scripts) and generated artifacts (`*.words.json`, `clean.wav`,
  `run.log`, `.venv/`) are gitignored — keep them out of commits.
