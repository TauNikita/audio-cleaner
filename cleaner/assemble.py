"""Cut the chosen spans and concatenate them with pauses into the clean output.

Each kept span is padded slightly so word edges aren't clipped, given a short fade at both ends so
the cuts don't click, and separated by a short silence between sentences / a longer one between
paragraphs.
"""

import os

from .align import Decision, LOW, MATCH
from .util import Progress, fmt_ts


def _load_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is not installed. Install it with:\n  pip install pydub"
        ) from exc
    return AudioSegment


def assemble(media: str, decisions: list[Decision], output: str,
             pad_ms: int, sentence_pause_ms: int, paragraph_pause_ms: int,
             crossfade_ms: int) -> None:
    """Render the kept spans to `output`. Format is inferred from the extension.

    Spans are cut from the original recording, not the bandwidth-limited 16kHz mono WAV used for
    transcription, so the output keeps the source sample rate and channels. Whisper's timestamps are
    in seconds, so they index the original media just as well.
    """
    AudioSegment = _load_pydub()
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
        start = max(0, int(round(d.start * 1000)) - pad_ms)
        end = min(total_ms, int(round(d.end * 1000)) + pad_ms)
        clip = audio[start:end]

        fade = min(crossfade_ms, len(clip) // 2)
        if fade > 0:
            clip = clip.fade_in(fade).fade_out(fade)

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
