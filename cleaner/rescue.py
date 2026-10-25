"""Second look at units the transcript could not place.

Whisper suppresses text it has just produced, and when a speaker re-reads a whole sentence three or
four times the full-file transcript can keep a single garbled copy of it ("It was the same thing"),
so the aligner finds no window at all (MISSING) or settles for a poor one (LOW). Such a unit must
still lie between its placed neighbours; hearing that stretch of audio in isolation shows every
reading, and the latest good one is the take to keep.
"""

from bisect import bisect_left, bisect_right

from .align import Decision, LOW, MATCH, MISSING, _candidates
from .hidden import Energy, Prober, _pick_restart
from .transcribe import Word
from .util import Progress, fmt_ts

WORDS_PER_S = 2.5          # unhurried narration; sizes the listening window to the unit
MIN_CHUNK_S = 20.0         # each listening window holds a whole reading of the unit with room to
CHUNK_FACTOR = 2.5         # spare, and consecutive windows overlap by more than one reading, so
STEP_MARGIN = 1.3          # every reading falls wholly inside some window
MIN_GAIN = 2.0             # a LOW unit's new placement must beat its old score by this much


def _bounds(decisions: list[Decision], k: int, duration: float) -> tuple[float, float]:
    """Audio between the placed neighbours of unit k."""
    a = 0.0
    for d in reversed(decisions[:k]):
        if d.status in (MATCH, LOW) and d.end is not None:
            a = d.end
            break
    b = duration
    for d in decisions[k + 1:]:
        if d.status in (MATCH, LOW) and d.start is not None:
            b = d.start
            break
    return a, b


def _word_range(words: list[Word], start: float, end: float) -> tuple[int, int]:
    """Transcript word indices covering [start, end] (may be empty)."""
    starts = [w.start for w in words]
    i = bisect_left(starts, start - 0.05)
    j = bisect_right(starts, end - 0.05)
    return i, max(i, j)


def rescue(decisions: list[Decision], words: list[Word], energy: Energy, prober: Prober,
           threshold: float, max_extra: int, low_delta: float = 12.0) -> int:
    """Re-place MISSING and LOW units from isolated hearings of the audio between their neighbours.
    Returns the number of units changed."""
    todo = [k for k, d in enumerate(decisions) if d.status in (MISSING, LOW) and not d.locked]
    if not todo:
        return 0
    progress = Progress("Rescue")
    low_cutoff = max(0.0, threshold - low_delta)
    changed = 0
    for n, k in enumerate(todo, 1):
        d = decisions[k]
        tokens = d.sentence.tokens
        a, b = _bounds(decisions, k, energy.duration)
        if not tokens or b - a < 0.5:
            continue
        expected = len(tokens) / WORDS_PER_S
        chunk = max(MIN_CHUNK_S, CHUNK_FACTOR * expected)
        step = chunk - STEP_MARGIN * expected
        found: list[tuple[float, float, float]] = []  # (start, end, score)
        t = max(0.0, a - 0.2)
        while True:
            heard = prober.words(energy, t, min(b + 0.2, t + chunk))
            for s, e, score in _candidates(heard, d.sentence, max_extra, low_cutoff):
                found.append((heard[s].start, heard[e - 1].end, score))
            if t + chunk >= b + 0.2:
                break
            t += step
        if not found:
            progress.update(f"{n}/{len(todo)} — #{d.sentence.index} still missing")
            continue
        good = [f for f in found if f[2] >= threshold]
        start, end, score = max(good, key=lambda f: f[0]) if good else max(found, key=lambda f: f[2])
        status = MATCH if good else LOW
        # A hearing that held two readings of the unit shows only the first: whisper suppresses the
        # second as repeated text, and the span then runs from a discarded attempt to the end of
        # the kept one. Hear the span again from past its opening: if the unit opens there anew,
        # that later reading is the take.
        while True:
            later = []
            off = 2.0
            while start + off < end - 0.6 * expected:
                inner = prober.words(energy, start + off, end + 0.3)
                later += [(inner[s].start, inner[e - 1].end, sc) for s, e, sc in
                          _candidates(inner, d.sentence, max_extra, low_cutoff)
                          if sc >= threshold - 4 and inner[s].start > start + 1.5]
                off += 3.0
            if not later:
                break
            start, end, sc2 = max(later, key=lambda f: f[0])
            score, status = max(score, sc2), MATCH if sc2 >= threshold else status
        if d.status == LOW and score < d.score + MIN_GAIN:
            progress.update(f"{n}/{len(todo)} — #{d.sentence.index} kept")
            continue
        # The hearing's timestamp for the first word jitters by a few tenths; snap the start to
        # the silence (or breath) before the reading, as a restart boundary is.
        start = _pick_restart([start], [], energy, floor=a) or start
        start = max(start, a)   # never before the previous unit's end: its tail would be heard twice
        was = "missing" if d.status == MISSING else f"low ({d.score:.0f})"
        d.start, d.end, d.score, d.status = start, end, score, status
        d.word_start, d.word_end = _word_range(words, start, end)
        d.runs = [(start, end)]
        d.excise = []
        d.notes.append(f"rescued: was {was}, heard in isolation at {fmt_ts(start)}-{fmt_ts(end)}")
        changed += 1
        progress.update(f"{n}/{len(todo)} — #{d.sentence.index} placed")
    progress.done(f"done — {changed}/{len(todo)} unplaced or doubtful units re-placed")
    return changed
