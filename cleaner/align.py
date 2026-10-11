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
from dataclasses import dataclass, field

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
    excise: list = field(default_factory=list)  # (start, end) seconds dropped inside the span
    notes: list = field(default_factory=list)   # what hidden.refine() changed, for the report
    runs: list | None = None  # (start, end) runs to cut, when the transcript words cannot say


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
# Consecutive units may overlap by this many transcript words; the later unit keeps them. This is
# how a window that ran a little past its unit's last word (fuzzy score tolerates a few extra
# words) gives those words back to the unit they belong to.
OVERLAP_WORDS = 3


def _best_window_at(words: list[Word], start: int, target: str,
                    lengths: range, n: int, cutoff: float) -> tuple[float, int]:
    """Best (score, end_index) over the candidate window lengths at a given start; on a tie the
    shorter window. Returns (0.0, start) when nothing reaches the cutoff.
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
    # best-fitting window of each take (earliest start on a tie, so a leading word is never shaved
    # off), so the placement below chooses between takes. A start that lies at or past the end of
    # the take's best window is a new take even when it follows directly: the unit said again
    # back to back, with nothing between the two readings.
    takes: list[list[tuple[int, int, float]]] = []
    best_end = -1
    for w in windows:
        if takes and w[0] - takes[-1][-1][0] <= TAKE_GAP and w[0] < best_end - OVERLAP_WORDS:
            takes[-1].append(w)
            if w[2] > max(x[2] for x in takes[-1][:-1]):
                best_end = w[1]
        else:
            takes.append([w])
            best_end = w[1]
    return [max(t, key=lambda w: (w[2], -w[0])) for t in takes]


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
        # Lookup tables: candidates sorted by end, with the running best dp value up to each end
        # and, at each position, the candidate that holds that running best (so a predecessor
        # looked up by end is the one whose value was used, not merely one that ends there).
        idx = sorted(range(len(cs)), key=lambda c: cs[c][1])
        ends.append([cs[c][1] for c in idx])
        run, holder, best_c = [], [], None
        for c in idx:
            if best_c is None or row[c] > row[best_c]:
                best_c = c
            run.append(row[best_c])
            holder.append(best_c)
        prefix.append(run)
        order.append(holder)
        progress.update(f"{k + 1}/{total} units placed")

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

    # A window may begin one word late when whisper rendered the words after the unit's first
    # word as copies of it ("another another another was" for "Another, Igor Melyantsev, was"):
    # the window skipping the real first word scores higher for carrying less garble. Reclaim a
    # first word that sits glued to the window's start and is not the previous unit's.
    from rapidfuzz import fuzz
    for i, k in enumerate(placed):
        s, e, sc = cands[k][chosen[k]]
        bound = cands[placed[i - 1]][chosen[placed[i - 1]]][1] if i else 0
        first = sentences[k].tokens[0]
        while s > bound and fuzz.ratio(words[s - 1].norm, first) >= 80 \
                and words[s].start - words[s - 1].end < 0.35:
            s -= 1
        cands[k][chosen[k]] = (s, e, sc)

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
