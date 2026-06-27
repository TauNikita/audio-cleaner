# audio-cleaner

Turn a messy voiceover recording into a clean read that follows a written script.

You record narration with pauses, false starts, and retakes. This tool transcribes the recording
with word-level timestamps, aligns each script sentence to the **last good take** of that sentence
in the audio, and stitches the chosen pieces together with natural pauses between sentences and
paragraphs.

Crucially, it lets you **review every pick before any audio is rendered** (`--dry-run`).

## How it works

```
recording (audio/video) ──ffmpeg──► 16kHz mono WAV
                                          │
                                  faster-whisper (word timestamps)
                                          │
script.txt ──► sentences ──► align each sentence to its last retake
                                          │
                          cut + concat with pauses ──► clean.wav
```

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
export LD_LIBRARY_PATH=$(uv run python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

# 4. Run on the GPU (float16 is selected automatically on cuda):
uv run clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" --device cuda
```

`--device auto` (the default) will pick the GPU on its own once the libraries above are in place. If
you hit a `cudnn`/`cublas` "library not found" error, the `LD_LIBRARY_PATH` export in step 3 is
almost always the fix. For lower VRAM use try `--compute-type int8_float16`.

### Setup (Windows)

The `gpu` extra has Windows wheels for cuBLAS + cuDNN 9, and the tool registers their DLL folders
automatically — so the pip/uv path works the same as on Linux, no manual PATH edits. NVIDIA driver
R570+ must be installed first (verify with `nvidia-smi`).

```powershell
# 1. Install the project plus the GPU runtime libraries (cuBLAS + cuDNN 9):
uv sync --extra gpu

# 2. Run on the GPU (float16 is selected automatically on cuda):
uv run clean.py "data\audio\Main.mp3" "data\script\Annexation of Crimea.txt" --device cuda
```

That's it — `--device auto` (the default) also picks the GPU once `--extra gpu` is installed.

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
| `--model` | `small.en` | faster-whisper model |
| `--language` | `en` | Spoken language |
| `--device` | `auto` | `auto` (GPU if available, else CPU), `cpu`, or `cuda` |
| `--compute-type` | auto | ctranslate2 compute type (float16 on GPU, int8 on CPU) |
| `--threshold` | `82` | Min fuzzy match score (0-100) to accept a take |
| `--max-extra` | `3` | Extra words to try beyond sentence length when matching |
| `--horizon` | unlimited | Max words to scan ahead per sentence |
| `--pad` | `60` | Padding (ms) added to each cut edge |
| `--sentence-pause` | `350` | Silence (ms) between sentences |
| `--paragraph-pause` | `900` | Silence (ms) between paragraphs |
| `--crossfade` | `15` | Crossfade (ms) at each join |
| `--cache` | auto | Transcript cache path |
| `--refresh` | off | Ignore the cache and re-transcribe |

## Notes

- Works best on a script of full sentences. Bullet points / shorthand match poorly (you'll get a
  warning).
- The score column in the dry-run makes a bad pick easy to spot — retakes are not assumed to be in
  order.
