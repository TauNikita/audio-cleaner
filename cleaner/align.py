"""Align every script unit (clause or sentence) to the spoken words, preferring the last retake.

Every unit is scored against candidate word windows across the whole transcript with a fuzzy
ratio. The units are then placed in script order by dynamic programming: each unit takes one
window, windows cannot overlap, later units come after earlier ones, and a unit that fits nowhere
is skipped (MISSING) at a fixed cost. Among takes that all read well (scores within a few points of
each other) the latest start wins, because the last attempt is the one the speaker meant to keep.

The order constraint is what makes short units safe: a clause that also occurs later in the same
sentence ("the Special Operations Forces" ... "their own Special Operations Forces Command") cannot
jump to the later occurrence, because the clauses after it would then have no room.
"""

from bisect import bisect_right
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


# Scores at or above threshold + GOOD_MARGIN are treated as equally good, so the later take wins;
# a take has to read clearly better than a later one to be preferred over it.
GOOD_MARGIN = 10.0
# Tie-break weight per word of transcript position (keeps "later wins" below any real score gap).
LATER_BONUS = 1e-4
# Cost of leaving a unit out. Larger than any score gain re-placing neighbours could buy.
SKIP_COST = 60.0
# How many consecutive units may be skipped in a row.
MAX_SKIP = 8
# Qualifying starts this close together belong to one take of the unit.
TAKE_GAP = 3
# Within a take, prefer the shortest window scoring within this many points of the best one, so a
# window does not swallow the next unit's first words for a point of fuzzy score.
TIGHT_MARGIN = 3.0
# Consecutive units may overlap by this many transcript words; the later unit keeps them.
OVERLAP_WORDS = 1


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


def _candidates(words: list[Word], unit: Sentence, max_extra: int,
                cutoff: float) -> list[tuple[int, int, float]]:
    """All (start, end, score) windows for one unit that clear the cutoff, in start order."""
    target = unit.norm
    if not target:
        return []
    n = len(words)
    length = len(unit.tokens)
    lengths = range(max(1, length - 2), length + max_extra + 1)
    windows = []
    for start in range(n):
        score, end = _best_window_at(words, start, target, lengths, n, cutoff)
        if score > 0.0:
            windows.append((start, end, score))
    # Neighbouring starts are the same take seen through slightly different windows; keep only the
    # best-fitting window of each take, so the placement below chooses between takes, not between
    # a tight window and one that swallows the next unit's first words.
    takes: list[list[tuple[int, int, float]]] = []
    for w in windows:
        if takes and w[0] - takes[-1][-1][0] <= TAKE_GAP:
            takes[-1].append(w)
        else:
            takes.append([w])
    picked = []
    for t in takes:
        best = max(w[2] for w in t)
        close = [w for w in t if w[2] >= best - TIGHT_MARGIN]
        picked.append(min(close, key=lambda w: (w[1] - w[0], -w[2], -w[0])))
    return picked


def align(words: list[Word], sentences: list[Sentence], threshold: float,
          max_extra: int, horizon: int | None, low_delta: float = 12.0) -> list[Decision]:
    """Place every unit in order; units that fit nowhere become MISSING."""
    low_cutoff = max(0.0, threshold - low_delta)
    good = threshold + GOOD_MARGIN
    progress = Progress("Align")
    total = len(sentences)

    cands: list[list[tuple[int, int, float]]] = []
    for k, unit in enumerate(sentences, 1):
        cands.append(_candidates(words, unit, max_extra, low_cutoff))
        progress.update(f"{k}/{total} units scored")

    # dp[k][c]: best total with unit k placed at its candidate c. back[k][c] = (j, c') predecessor.
    dp: list[list[float]] = []
    back: list[list[tuple[int, int] | None]] = []
    ends: list[list[int]] = []          # candidate ends per unit, sorted (candidates are in start
    prefix: list[list[float]] = []      # order; we sort by end for the lookup below)
    order: list[list[int]] = []
    for k, cs in enumerate(cands):
        row, brow = [], []
        for start, end, score in cs:
            value = min(score, good) + LATER_BONUS * start
            best_prev, best_from = -SKIP_COST * k, None  # everything before skipped
            for j in range(k - 1, max(-1, k - 2 - MAX_SKIP), -1):
                if not cands[j]:
                    continue
                pos = bisect_right(ends[j], start + OVERLAP_WORDS) - 1
                if pos < 0:
                    continue
                v = prefix[j][pos] - SKIP_COST * (k - 1 - j)
                if v > best_prev:
                    best_prev, best_from = v, (j, order[j][pos])
            row.append(value + best_prev)
            brow.append(best_from)
        dp.append(row)
        back.append(brow)
        # Lookup tables: candidates sorted by end, with the running best dp value up to each end.
        idx = sorted(range(len(cs)), key=lambda c: cs[c][1])
        order.append(idx)
        ends.append([cs[c][1] for c in idx])
        run, best = [], float("-inf")
        for c in idx:
            best = max(best, row[c])
            run.append(best)
        prefix.append(run)
        progress.update(f"{k + 1}/{total} units placed")
    # prefix[] holds running maxima, so recover the exact candidate behind each lookup value.
    for j in range(len(cands)):
        best_c, best_v = None, float("-inf")
        fixed = []
        for pos, c in enumerate(order[j]):
            if dp[j][c] > best_v:
                best_v, best_c = dp[j][c], c
            fixed.append(best_c)
        order[j] = fixed

    # Best finish: the last placed unit, with any trailing units skipped.
    best_total, best_at = -SKIP_COST * total, None
    for k in range(total):
        for c, v in enumerate(dp[k]):
            if v - SKIP_COST * (total - 1 - k) > best_total:
                best_total, best_at = v - SKIP_COST * (total - 1 - k), (k, c)

    chosen: dict[int, int] = {}
    cur = best_at
    while cur is not None:
        k, c = cur
        chosen[k] = c
        cur = back[k][c]

    # Where two placed units overlap by a word, the later unit keeps it.
    placed = sorted(chosen)
    for a, b in zip(placed, placed[1:]):
        s0, e0, sc0 = cands[a][chosen[a]]
        s1 = cands[b][chosen[b]][0]
        if e0 > s1 > s0:
            cands[a][chosen[a]] = (s0, s1, sc0)

    decisions: list[Decision] = []
    for k, unit in enumerate(sentences):
        if k in chosen:
            start, end, score = cands[k][chosen[k]]
            decisions.append(Decision(
                sentence=unit, status=MATCH if score >= threshold else LOW, score=score,
                start=words[start].start, end=words[end - 1].end,
                word_start=start, word_end=end,
            ))
        else:
            decisions.append(Decision(sentence=unit, status=MISSING, score=0.0,
                                      start=None, end=None, word_start=None, word_end=None))
    matched = sum(1 for d in decisions if d.status in (MATCH, LOW))
    progress.done(f"done — {matched}/{total} matched")
    return decisions
