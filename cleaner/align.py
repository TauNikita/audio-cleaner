"""Align each script sentence to the spoken words, preferring the last retake.

For every sentence we scan the transcript forward from a moving pointer, score candidate word
windows against the sentence with a fuzzy ratio, and keep the qualifying window with the *latest*
start. Latest-start is what picks the final retake over earlier flubbed attempts. The pointer only
advances past a matched span, so each sentence is searched only in audio after the previous match.
"""

from dataclasses import dataclass

from .text import Sentence
from .transcribe import Word
from .util import Progress

MATCH = "MATCH"
LOW = "LOW"          # only cleared the relaxed retry threshold
MISSING = "MISSING"  # nothing cleared even the relaxed threshold


@dataclass
class Decision:
    sentence: Sentence
    status: str
    score: float
    start: float | None       # seconds
    end: float | None         # seconds
    word_start: int | None    # index into the word list (inclusive)
    word_end: int | None      # index into the word list (exclusive)


def _best_window_at(words: list[Word], start: int, target: str,
                    lengths: range, n: int, cutoff: float) -> tuple[float, int]:
    """Best (score, end_index) over the candidate window lengths at a given start.

    Returns (0.0, start) when nothing reaches the cutoff.
    """
    from rapidfuzz import fuzz

    best_score = 0.0
    best_end = start
    for length in lengths:
        end = start + length
        if end > n:
            break
        window = " ".join(words[i].norm for i in range(start, end))
        score = fuzz.ratio(target, window, score_cutoff=cutoff)
        if score > best_score:
            best_score = score
            best_end = end
    return best_score, best_end


# How far apart two qualifying start positions can be and still count as the same take. Within one
# spoken take, many consecutive starts qualify; a real retake is separated by a flubbed/again gap.
_CLUSTER_GAP = 3


def _pick_last_take(cands: list[tuple[int, int, float]]) -> tuple[int, int, float]:
    """From qualifying (start, end, score) windows, pick the best window of the latest take.

    Grouping consecutive starts into takes, then choosing the highest score within the latest take,
    gives us the final retake without the latest-start rule shaving the first word off a sentence.
    """
    last_take = [cands[0]]
    for cand in cands[1:]:
        if cand[0] - last_take[-1][0] <= _CLUSTER_GAP:
            last_take.append(cand)
        else:
            last_take = [cand]  # a gap means a new take started; keep only the newest
    # Best score in the latest take; tie-break on the later start (shorter, tighter window).
    return max(last_take, key=lambda c: (c[2], c[0]))


def align_sentence(words: list[Word], pointer: int, sentence: Sentence,
                   threshold: float, max_extra: int, horizon: int | None,
                   low_delta: float) -> tuple[float, int, int, str]:
    """Find the best span for one sentence.

    Returns (score, word_start, word_end, status). word_start is -1 when MISSING.
    """
    target = sentence.norm
    if not target:
        return 0.0, -1, -1, MISSING

    n = len(words)
    length = len(sentence.tokens)
    lengths = range(max(1, length - 2), length + max_extra + 1)
    low_cutoff = max(0.0, threshold - low_delta)

    scan_end = n if horizon is None else min(n, pointer + horizon)

    high: list[tuple[int, int, float]] = []
    low: list[tuple[int, int, float]] = []
    for start in range(pointer, scan_end):
        score, end = _best_window_at(words, start, target, lengths, n, low_cutoff)
        if score <= 0.0:
            continue
        (high if score >= threshold else low).append((start, end, score))

    if high:
        start, end, score = _pick_last_take(high)
        return score, start, end, MATCH
    if low:
        start, end, score = _pick_last_take(low)
        return score, start, end, LOW
    return 0.0, -1, -1, MISSING


def align(words: list[Word], sentences: list[Sentence], threshold: float,
          max_extra: int, horizon: int | None, low_delta: float = 12.0) -> list[Decision]:
    """Align every sentence in order. Pointer advances only past matched spans."""
    decisions: list[Decision] = []
    pointer = 0
    progress = Progress("Align")
    total = len(sentences)

    for i, sentence in enumerate(sentences, 1):
        score, w_start, w_end, status = align_sentence(
            words, pointer, sentence, threshold, max_extra, horizon, low_delta
        )
        if status in (MATCH, LOW):
            decisions.append(Decision(
                sentence=sentence, status=status, score=score,
                start=words[w_start].start, end=words[w_end - 1].end,
                word_start=w_start, word_end=w_end,
            ))
            pointer = w_end  # only matched sentences move the pointer
        else:
            decisions.append(Decision(
                sentence=sentence, status=MISSING, score=0.0,
                start=None, end=None, word_start=None, word_end=None,
            ))
        progress.update(f"{i}/{total} sentences — pointer at word {pointer}")

    matched = sum(1 for d in decisions if d.status in (MATCH, LOW))
    progress.done(f"done — {matched}/{total} matched")
    return decisions
