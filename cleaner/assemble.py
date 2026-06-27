"""Cut the chosen spans and concatenate them with pauses into the clean output.

Each kept span is padded slightly so word edges aren't clipped, given a short fade at both ends so
the cuts don't click, and separated by a short silence between sentences / a longer one between
paragraphs. Large time gaps *inside* a span are excised (see _split_runs) so dead air and dropped
retakes don't make it into the output.
"""

import os

from .align import Decision, LOW, MATCH
from .transcribe import Word
from .util import Progress, fmt_ts


# A short, controlled pause that replaces each excised internal gap, so a stitched span doesn't
# sound abruptly spliced where dead air / a flubbed retake was cut out.
_GAP_PAUSE_MS = 200

# Whisper's word-end timestamp marks where a word's energy mostly ends, clipping the natural release.
# At a sentence's true end we fade out over a longer window than the click-killing crossfade so the
# voice tapers naturally into the pause instead of stopping abruptly.
_END_FADE_MS = 150


def _split_runs(words: list[Word], w_start: int, w_end: int,
                max_gap_s: float) -> list[tuple[float, float]]:
    """Split a kept word span into temporally-contiguous runs of (start, end) seconds.

    Whisper's chosen words are contiguous by index but not always by time: when vad_filter drops a
    flubbed retake, a few stray words straddle it with a large time gap. Cutting [first.start,
    last.end] as one block would re-include that retake / dead audio — the source of output repeats —
    so we break the span wherever the gap between consecutive words exceeds max_gap_s and drop the
    gap. (A single word with a stretched timestamp spanning the gap is a residual case this doesn't
    fully catch.)
    """
    runs: list[tuple[float, float]] = []
    run_start = words[w_start].start
    prev_end = words[w_start].end
    for wi in range(w_start + 1, w_end):
        w = words[wi]
        if w.start - prev_end > max_gap_s:
            runs.append((run_start, prev_end))
            run_start = w.start
        prev_end = w.end
    runs.append((run_start, prev_end))
    return runs


def _load_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is not installed. Install it with:\n  pip install pydub"
        ) from exc
    return AudioSegment


def assemble(media: str, decisions: list[Decision], words: list[Word], output: str,
             pad_ms: int, tail_pad_ms: int, sentence_pause_ms: int, paragraph_pause_ms: int,
             crossfade_ms: int, max_internal_gap_ms: int) -> None:
    """Render the kept spans to `output`. Format is inferred from the extension.

    Spans are cut from the original recording, not the bandwidth-limited 16kHz mono WAV used for
    transcription, so the output keeps the source sample rate and channels. Whisper's timestamps are
    in seconds, so they index the original media just as well.

    Each span is cut from its temporally-contiguous runs (see _split_runs), excising internal gaps
    longer than max_internal_gap_ms so dead air and dropped retakes don't end up in the output. The
    lead edge gets pad_ms; the sentence's true end gets the larger tail_pad_ms plus a longer fade so
    the natural decay isn't clipped.
    """
    AudioSegment = _load_pydub()
    max_gap_s = max_internal_gap_ms / 1000.0
    # Decode the original media so cuts keep full fidelity; let pydub/ffmpeg detect the format.
    audio = AudioSegment.from_file(media)
    frame_rate = audio.frame_rate
    total_ms = len(audio)

    kept = [d for d in decisions if d.status in (MATCH, LOW)]
    if not kept:
        raise RuntimeError("nothing to render: no sentences matched")

    progress = Progress("Assemble")
    result = AudioSegment.empty()
    for i, d in enumerate(kept):
        # Don't let a padded sentence end bleed into the next sentence's (also lead-padded) start.
        next_start_ms = (int(round(kept[i + 1].start * 1000)) - pad_ms
                         if i + 1 < len(kept) else total_ms)

        # Build the span from its contiguous runs, dropping large internal gaps. The lead edge gets
        # pad_ms; the sentence's true end gets the larger tail_pad_ms and a longer fade so the natural
        # decay tapers instead of stopping abruptly. Internal splice edges get only the anti-click
        # crossfade.
        runs = _split_runs(words, d.word_start, d.word_end, max_gap_s)
        clip = AudioSegment.empty()
        for r, (run_start, run_end) in enumerate(runs):
            is_last = r == len(runs) - 1
            run_end_ms = int(round(run_end * 1000))
            start = int(round(run_start * 1000)) - (pad_ms if r == 0 else 0)
            if is_last:
                end = min(run_end_ms + tail_pad_ms, max(run_end_ms, next_start_ms))
            else:
                end = run_end_ms
            sub = audio[max(0, start):min(total_ms, end)]

            half = len(sub) // 2
            fade_in = min(crossfade_ms, half)
            fade_out = min(_END_FADE_MS if is_last else crossfade_ms, half)
            if fade_in > 0:
                sub = sub.fade_in(fade_in)
            if fade_out > 0:
                sub = sub.fade_out(fade_out)

            if r > 0 and _GAP_PAUSE_MS > 0:
                gap_pause = AudioSegment.silent(duration=_GAP_PAUSE_MS, frame_rate=frame_rate)
                clip += gap_pause.set_channels(audio.channels)
            clip += sub

        if i > 0:
            prev = kept[i - 1]
            pause = paragraph_pause_ms if prev.sentence.paragraph_end else sentence_pause_ms
            if pause > 0:
                silence = AudioSegment.silent(duration=pause, frame_rate=frame_rate)
                result += silence.set_channels(audio.channels)

        result += clip
        progress.update(f"{i + 1}/{len(kept)} spans — output {fmt_ts(len(result) / 1000)}")

    ext = os.path.splitext(output)[1].lstrip(".").lower() or "wav"
    progress.update(f"encoding {output} ({ext})", force=True)
    result.export(output, format=ext)
    progress.done(f"wrote {output} — {fmt_ts(len(result) / 1000)} from {len(kept)} spans")
