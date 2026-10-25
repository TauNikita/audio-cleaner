# audio-cleaner

Turn a messy voiceover recording into a clean read that follows a written script.

You record narration with pauses, false starts, and retakes. This tool transcribes the recording
with word-level timestamps, aligns each script sentence to the **last good take** of that sentence
in the audio, cuts every take at its audible edges and stitches the pieces together with pauses of
a fixed, audible length between sentences and paragraphs. `--clauses` aligns at commas, semicolons,
colons and dashes instead, which picks up a retake from the last comma but adds cuts inside sentences.

Crucially, it lets you **review every pick before any audio is rendered** (`--dry-run`).

## How it works

```
recording (audio/video) ──ffmpeg──► 16kHz mono WAV
                                          │
                                  faster-whisper (word timestamps)
                                          │
script.txt ──► sentences ──► place every sentence, in order, on its last good take
                                          │
                   listen again inside each pick for retakes whisper hid
                                          │
                 cut at the audible edges + concat with pauses ──► clean.wav (+ clean.cuts.json)
                                          │
                     --verify: re-transcribe the output, sentence by sentence
```

Whisper suppresses text it has just produced, so a sentence restarted mid-flow often transcribes as
one long word or as nothing at all — and the aligner would keep the whole stretch, so the output
repeats. The refine pass re-transcribes short pieces of audio in isolation, starting *inside* those
spots, and keeps only the last attempt. A sentence the transcript lost altogether (re-read so often
that whisper kept one garbled copy) is rescued by hearing the audio between its neighbours in
isolation. Finally no cut may fall inside a word: a cut that lands in a word's sound is moved back
to where that word begins. Repeats are judged against the script: words the script itself says twice
("the best of the best") are not a retake, and a cut never removes words heard nowhere else.
`--verify` then checks every sentence of the
rendered file against the script and prints the ones worth a listen, with output timestamps.

GPU transcription is not bit-for-bit deterministic, so two runs can place a cut a few tens of
milliseconds apart; the boundaries are snapped to silences (or the breath between two readings) so
that this jitter never lands inside a word.

## Edges and pauses

Whisper's timestamps are not where the sound is: a word's end timestamp comes a median 180 ms
before its sound has died away, and a first word's start lands early or late by a hundred ms or
more. Cutting at the timestamps clips word endings and leaves uneven dead air. So every piece is
cut at its **audible edges**, measured on the 16 kHz track against the recording's noise floor:

- the tail runs on from the last word until the sound has settled for good — a quiet of at least
  180 ms, longer than the closure before a final "t" or the dip before a trailing "s" (those stay
  under ~120 ms) — and keeps `--tail` (80 ms) of room after it, so the decay is never clipped; a
  last-word timestamp that already sits in the quiet (the VAD's padding runs long) is walked back to
  the sound it belongs to;
- the lead keeps `--lead` (100 ms) of room before the first sound that is speech — a breath or a
  click a few dB over the floor is not —: an early timestamp's dead air (up to 1.2 s of VAD padding)
  is trimmed, a late one is moved back to the word's start when a clean onset is there; a "word" of a
  few ms at a segment start with no sound anywhere near it is a transcript artifact and is dropped;
- a **pause is what you hear**: `--sentence-pause` / `--paragraph-pause` (600 ms) and
  `--clause-pause` (300 ms) are the silence from the last sound of one piece to the first sound of
  the next. The room the pieces kept counts towards it and only the remainder is inserted, so every
  pause comes out the same length whatever the takes' own room was.

A silence inside a sentence longer than `--max-inner-pause` (700 ms, the speaker finding the line)
is shortened to the clause pause. Where a retake was cut out of the middle of a sentence, or where
the stretches of a hand-placed unit (`--overrides`) end, speech may run straight through the join;
such a cut lands at the quietest moment next to the boundary with a `--hard-fade` (40 ms), and never
reaches back into the stretch that was cut out. No audio is ever heard twice: a piece whose
timestamp falls inside the previous piece's cut starts after that cut. Every piece is logged in `<output>.cuts.json` (source and output times, the
edges found, the room kept, the silence inserted) so the joins can be audited or listened to.

## From scratch on a new machine

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment.

### 1. Install ffmpeg (system package)

```bash
# Debian / Ubuntu
sudo apt install ffmpeg
# macOS (Homebrew)
brew install ffmpeg
```

```powershell
# Windows (PowerShell) — pick one
winget install Gyan.FFmpeg
# or:  choco install ffmpeg
# or:  scoop install ffmpeg
```

After installing on Windows, `ffmpeg` won't be visible in the **current** terminal until `PATH`
refreshes. Either open a new terminal, or reload `PATH` in place:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
ffmpeg -version    # should print a version
```

### 2. Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# then restart the shell, or:  export PATH="$HOME/.local/bin:$PATH"
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# or:  winget install astral-sh.uv
```

### 3. Get the code and create the environment

```bash
git clone <repo-url> audio-cleaner
cd audio-cleaner
uv sync                 # creates .venv and installs dependencies from pyproject.toml / uv.lock
```

`uv` reads `pyproject.toml`, creates `.venv`, and installs everything. No manual venv or
`pip install` needed. `uv sync` is reproducible from the committed `uv.lock`.

### 4. Run it

`uv run` executes inside the managed environment (no need to activate anything).

Review the picks first — this writes **no audio**:

```bash
uv run clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" --dry-run
```

When the report looks right, render:

```bash
uv run clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" -o clean.wav
```

The transcript is cached next to the media (`*.words.json`), so re-running the dry-run while you
tune `--threshold` is instant. Use `--refresh` to force re-transcription.

The same `uv run clean.py ...` commands work on Windows (PowerShell); paths may use `\` or `/`.

> Prefer plain pip? Linux/macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
> run with `.venv/bin/python clean.py ...`. Windows: `py -m venv .venv; .venv\Scripts\pip install -r
> requirements.txt`, run with `.venv\Scripts\python clean.py ...`. The uv path above is recommended.

## GPU acceleration (NVIDIA, incl. RTX 50-series / RTX 5070)

By default `--device auto` uses the GPU when one is visible and falls back to CPU otherwise. On CPU,
transcription runs at roughly real time; a GPU is dramatically faster.

faster-whisper runs on [CTranslate2](https://github.com/OpenNMT/CTranslate2) (not PyTorch), so GPU
use only needs the NVIDIA driver plus the CUDA 12 cuBLAS / cuDNN 9 runtime libraries.

**Blackwell / RTX 50-series notes (RTX 5070, 5080, 5090):** these are compute capability `sm_120`
and need a **CUDA 12.8+ driver (R570 or newer)** — check with `nvidia-smi`. The bundled
`ctranslate2` (>= 4.5, this project uses 4.8) targets CUDA 12 + cuDNN 9, which is what these cards
want.

### Setup (Linux)

```bash
# 1. NVIDIA driver R570+ must already be installed (verify: nvidia-smi)

# 2. Install the project plus the GPU runtime libraries (cuBLAS + cuDNN 9):
uv sync --extra gpu

# 3. ctranslate2 finds the pip-installed CUDA libraries via LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=$(uv run --extra gpu python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

# 4. Run on the GPU (float16 is selected automatically on cuda):
uv run --extra gpu clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" --device cuda
```

**Use `uv run --extra gpu` every run.** Plain `uv run` re-syncs the environment to the default
dependencies and removes the GPU libraries, so the next run would fail to find cuBLAS/cuDNN. Passing
`--extra gpu` keeps them installed.

`--device auto` (the default) picks the GPU on its own once the libraries are in place. If you hit a
`cudnn`/`cublas` "library not found" error, the `LD_LIBRARY_PATH` export in step 3 is almost always
the fix. For lower VRAM use `--compute-type int8_float16`.

### Setup (Windows)

The `gpu` extra has Windows wheels for cuBLAS + cuDNN 9, and the tool registers their DLL folders
automatically — so the pip/uv path works the same as on Linux, no manual PATH edits. NVIDIA driver
R570+ must be installed first (verify with `nvidia-smi`).

```powershell
# 1. Install the project plus the GPU runtime libraries (cuBLAS + cuDNN 9):
uv sync --extra gpu

# 2. Run on the GPU (float16 is selected automatically on cuda):
uv run --extra gpu clean.py "data\audio\Main.mp3" "data\script\Annexation of Crimea.txt" --device cuda
```

**Use `uv run --extra gpu` every run.** Plain `uv run` re-syncs to the default dependencies and
removes the GPU libraries, so the next run would fail to find cuBLAS/cuDNN. Passing `--extra gpu`
keeps them installed. `--device auto` (the default) also picks the GPU once they're present.

If you still hit a `cublas`/`cudnn` "not found" error, the alternative is Purfview's prebuilt
NVIDIA libraries (https://github.com/Purfview/whisper-standalone-win, "CUDA libs"): unzip and put
the DLLs on `PATH` or next to `clean.py`. For lower VRAM use `--compute-type int8_float16`.

> A harmless `huggingface_hub` symlink warning may appear on first run (model download). To silence
> it, enable Windows Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. It does not affect
> results.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-o, --output` | `clean.wav` | Output file (format inferred from extension) |
| `--dry-run` | off | Align and print the report, write no audio |
| `--clauses` | off | Align clauses instead of whole sentences (more cuts; may clip word endings) |
| `--no-refine` | off | Skip the hidden-retake pass (re-listening inside each pick) |
| `--verify` | off | After rendering, re-transcribe each sentence from the output and flag mismatches |
| `--overrides` | none | JSON `{"63": {"runs": [[1436.2, 1437.5], [1444.15, 1447.95]]}}`: cut unit 63 from exactly these stretches of the recording (seconds), skipping the listening passes for it |
| `--model` | `small.en` | faster-whisper model |
| `--language` | `en` | Spoken language |
| `--device` | `auto` | `auto` (GPU if available, else CPU), `cpu`, or `cuda` |
| `--compute-type` | auto | ctranslate2 compute type (float16 on GPU, int8 on CPU) |
| `--threshold` | `82` | Min fuzzy match score (0-100) to accept a take |
| `--max-extra` | `3` | Extra words to try beyond sentence length when matching |
| `--horizon` | unlimited | Max words to scan ahead per sentence |
| `--sentence-pause` | `600` | Pause heard between sentences (ms): last sound to first sound |
| `--paragraph-pause` | `600` | Pause heard between paragraphs (ms) |
| `--clause-pause` | `300` | Pause heard inside a sentence (ms): between clauses (`--clauses`), at a shortened inner pause, at a join where a retake was cut out |
| `--lead` | `100` | Room kept before a piece's first sound (ms) |
| `--tail` | `80` | Room kept after a piece's last sound has died down (ms) |
| `--tail-max` | `900` | How far past the last word's timestamp the sound may run on before the cut is forced (ms) |
| `--max-inner-pause` | `700` | A silence inside a sentence longer than this (ms) is shortened to the clause pause; `0` keeps every inner pause |
| `--max-internal-gap` | `1200` | A gap between consecutive transcript words longer than this (ms) splits the span, dropping the dead air or retake in between |
| `--hard-fade` | `40` | Fade (ms) at a cut that has to land in running speech |
| `--cache` | auto | Transcript cache path |
| `--refresh` | off | Ignore the cache and re-transcribe |

## Notes

- Works best on a script of full sentences. Bullet points / shorthand match poorly (you'll get a
  warning).
- The score column in the dry-run makes a bad pick easy to spot — retakes are not assumed to be in
  order. In the `P` column, `,` marks a clause that continues its sentence, `¶` a paragraph end.
- `AUDIO_CLEANER_EXPLAIN=12,40 uv run clean.py ... --dry-run` prints, for units 12 and 40, every
  probe the hidden-retake pass heard and the restart evidence it drew from them.
