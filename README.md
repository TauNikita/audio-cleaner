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

## Requirements

- **ffmpeg** on the system: `sudo apt install ffmpeg`
- Python deps (in a virtualenv, since system Python is usually locked down):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Always run inside the venv (e.g. `.venv/bin/python` or after `source .venv/bin/activate`).

Review the picks first — this writes **no audio**:

```bash
.venv/bin/python clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" --dry-run
```

When the report looks right, render:

```bash
.venv/bin/python clean.py "data/audio/Main.mp3" "data/script/Annexation of Crimea.txt" -o clean.wav
```

The transcript is cached next to the media (`*.words.json`), so re-running the dry-run while you
tune `--threshold` is instant. Use `--refresh` to force re-transcription.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-o, --output` | `clean.wav` | Output file (format inferred from extension) |
| `--dry-run` | off | Align and print the report, write no audio |
| `--model` | `small.en` | faster-whisper model |
| `--language` | `en` | Spoken language |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--compute-type` | `int8` | ctranslate2 compute type |
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
