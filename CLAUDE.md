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
                  rescue.rescue(): hear the gap around a unit the transcript could not place
                                          │
                  hidden.refine(): listen again inside each pick, drop hidden retakes
                                          │
                          cut + concat with pauses ──► clean.wav
                                          │
                     verify.verify(): re-transcribe the output per sentence (--verify)
```

## Module map (`cleaner/`)

| File | Responsibility |
| --- | --- |
| `text.py` | `normalize()` (shared by script + transcript), `split_script()` into units: whole sentences by default, clauses with `clauses=True` (`--clauses`: split at `, ; :` and spaced dashes, pieces under 4 words folded into a neighbour; opt-in because the extra cuts can clip word endings). Each unit carries `sentence_end` / `paragraph_end` for the pauses. Sentence splitting guards decimals/abbreviations. |
| `audio.py` | `extract_wav()` — ffmpeg to 16kHz mono temp WAV, with progress. `ensure_ffmpeg()`, `DependencyError`. |
| `transcribe.py` | faster-whisper word timestamps; JSON cache keyed on media+model+language; `resolve_device()` for auto/cpu/cuda + compute-type defaults. |
| `align.py` | **Core algorithm.** Every unit is scored against windows over the whole transcript, one window per take; a DP then places all units in order (no overlaps beyond one word, a unit that fits nowhere is skipped at a fixed cost), preferring the later take among takes that read equally well. See below. `Decision.excise` / `.notes` are filled by `hidden.refine()`. |
| `hidden.py` | **Hidden-retake pass.** Whisper suppresses repeated text, so a restart inside a pick shows up as a stretched word or a blank, not as words. `Energy` (20 ms loudness mask of the 16 kHz WAV), `Prober` (re-transcribe a slice in isolation, numpy in), `refine()`. Evidence rules: a repeat is trusted only when one probe hears the script words twice (`_repeats`), or when a probe starting where the run's own hearing went blank hears the sentence opening (`_opening_run` + `_blank`); cross-probe word comparisons are never used because whisper's timestamps drift between probes. A mid-sentence repeat needs a content word said twice (two function words around a garble are not evidence), and a fuzzy re-hearing of the opening must not be the opening shifted by a word or two (the sentence carrying on). The restart boundary snaps to a real silence, failing that to the breath between two readings, never inside the first word. Last, a cut must not fall inside a word: when the cut lies inside, or in a short dip just after, a run of loud frames that began within a second (`Energy.loud_run_start`, loudness relative to the word so breaths and room noise do not count), it moves to the run's start, provided a hearing from there opens on the first word or a garble of it, not on the previous unit's last words, and does not say the opening twice. Hearings alone are not trusted for this: whisper drops a short first word that begins right at a slice's start. An excision never removes script words heard nowhere else (`_excision_safe`); words the script itself says twice are not a repeat; a gap right after a stretched transcript word is not split out (`assemble._split_runs`: whisper collapsed speech into that word); and a span whose end whisper collapsed is extended to where the sentence is heard whole. Also fixes a late first-word timestamp (`onset_before`); `Energy.pcm` is the track `assemble` builds its own 10 ms `_Fine` from. Two more hearings per span: the speech inside a gap between two of its runs (a gap over `max_gap_s` with sound in it is often a whole re-reading whisper suppressed as repeated text, which would glue the first attempt's opening to the retake's end): if the unit opens again there, the span is cut whole from that restart (`Decision.runs = [(restart, end)]`); and a later run whose first transcript word is heard before its timestamp inside running speech (whisper timing a run's first words late) starts where it is heard (`run starts heard earlier`). |
| `rescue.py` | **Second look.** A sentence re-read three or four times can survive in the transcript as one garbled copy, so the aligner finds no window (MISSING) or a poor one (LOW). The unit must lie between its placed neighbours: hear that stretch in isolation (windows of 2.5x the unit's expected length, overlapping by more than one reading), run `align._candidates` on the hearing and keep the latest good take. Sets `Decision.runs` so assemble cuts the heard span, not the transcript's words. |
| `verify.py` | `--verify`: re-transcribe each kept sentence from the rendered output in isolation, compare with the script (fuzzy ratio, extra words, duration vs the speaker's own pace, stretched words) and print the rows worth a listen with output timestamps. |
| `report.py` | Dry-run table + MISSING / low-confidence flags + fix hints. |
| `assemble.py` | Cuts every piece at its **audible edges** and joins the pieces with pauses of a fixed audible length. `_Fine`: 10 ms loudness of the 16 kHz track (`Energy.pcm`), sound = > noise floor (5th percentile) + 10 dB with loud blips < 40 ms dropped, soft = > floor + 4 dB. `speech()`: the sound mask with runs that cannot be words set quiet (peak > 15 dB under the piece's speech level — breaths sit 20–26 dB under — unless the run lasts ≥ 100 ms within 24 dB: a soft first word like "In"; or < 60 ms and alone). A breath in the pre-roll is room and stays; a cut in a quiet dip glued to the word gets a 12 ms fade, not the long one. `audible_end()`: the sentence's speech has ended at the first quiet of ≥ 180 ms (or ≥ 130 ms followed by ≥ 150 ms of speech) — longer than a stop closure or the dip before a trailing fricative; a timestamp already in the quiet is walked back (≤ 1.5 s) to the speech it belongs to. `audible_onset()`: the first speech after an early timestamp (≤ 1.2 s ahead: the VAD pads 400–800 ms) or the clean start of the speech a late timestamp sits in (≤ 700 ms back, ≥ 150 ms of quiet before it), else None → the cut falls at the quietest moment (speech glued to earlier speech). `_drop_silent()`: a piece with no sound from 100 ms before to 500 ms after its timestamps is a transcript artifact (a 10 ms "word" pinned to a segment start) and is dropped. `_edges()`: lead = `--lead` of room before the onset (never into earlier sound or the previous piece's cut), tail = `--tail` of room after the audible end (capped by `--tail-max`, the next piece's words, and `pb + 50 ms` at a hard edge); when the speech never settles (a take switch inside running speech) the cut falls at the quietest moment next to the word boundary; fades land in the room, a cut forced into sound gets `--hard-fade`. Pauses are the audible gap: inserted silence = pause − room kept on both sides. `_clip_runs` marks excision edges hard; `_split_inner` breaks a piece at quiet stretches ≥ `--max-inner-pause` (joined with the clause pause). Writes `<output>.cuts.json` (every piece: source/output times, edges, room, silence) for audits. pydub for cutting/export (format from extension). |
| `cli.py` | `--overrides JSON` places named units by hand (explicit runs in seconds; `Decision.locked` keeps refine/rescue off them) for the rare take no heuristic can split, such as a sentence whose only complete reading is the first two words of one attempt plus the rest of another. argparse, lazy heavy-imports (so `--help` works with nothing installed), orchestration, temp cleanup. |
| `util.py` | `fmt_ts()`, `Progress` (single-line progress for long stages). |

`clean.py` is the thin entry point (`python clean.py ...` / `uv run clean.py ...`).

## The alignment core (`align.py`)

- For each unit, score word windows (lengths `len-2 .. len+max_extra`) at every transcript position
  with `rapidfuzz.fuzz.ratio`; at one start the best-scoring length wins (shorter on a tie).
  Qualifying starts within `TAKE_GAP` of each other are one **take** unless a start lies at or past
  the end of the take's best window (the unit said again back to back); a take keeps its
  best-scoring window, earliest start on a tie, so a leading word is never shaved off.
- Place all units by dynamic programming in script order: windows may overlap by up to
  `OVERLAP_WORDS` (the later unit keeps the words, which is how a window that ran a few words past
  its unit gives them back), skipping a unit costs `SKIP_COST`, and the value of a take is
  `min(score, threshold + GOOD_MARGIN)` plus a tiny later-is-better bonus. So among takes that all
  read well the **latest** wins (the final retake), while a clearly worse take loses. The end-sorted
  lookup tables record, per position, the candidate that holds the running best, so trace-back
  returns the candidate whose value was used.
- The order constraint is what makes short units safe: a clause that recurs later in its sentence
  cannot jump there, because the clauses after it would have no room.
- Scores below `--threshold` but above the relaxed cutoff → `LOW`; a unit that fits nowhere → `MISSING`
  (never crash).

## Commands

Environment is managed with **uv**.

```bash
uv sync                                   # create .venv + install deps
uv run clean.py MEDIA SCRIPT --dry-run    # review picks, write no audio
uv run clean.py MEDIA SCRIPT -o clean.wav # render (hidden-retake pass on by default; --no-refine to skip)
uv run clean.py MEDIA SCRIPT -o clean.wav --verify   # ...and check every output sentence against the script
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
