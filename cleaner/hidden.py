"""Find and drop retakes that whisper's transcript hides inside a chosen span.

Whisper suppresses text it has just emitted, so when a sentence is restarted mid-flow
("...Kvachkov was placed | While Girkin was ... Kvachkov was placed under...") the second attempt
often transcribes as one stretched word, or as nothing at all. The aligner then keeps the whole
stretch and the output repeats. The transcript can't show this, but the audio can: a single word
lasting over a second, or an inter-word gap that is not actually silent. At each such spot we
re-transcribe the audio in isolation, starting *inside* the spot so there is no prior text to
suppress against. Each probe's words are matched to the script as runs. A run landing on script
words the same probe already heard earlier is the same words said again: a restart if it is the
sentence opening (everything before it is dropped), a stutter otherwise (the first saying is cut).
A probe that starts where the span's own hearing went blank and hears the opening there is also a
restart. Nothing else is trusted, because whisper's timestamps drift between probes.

The same energy track fixes two edge problems: a first word whose timestamp lands late (the span
is extended back to the real speech onset) and a sentence tail that would run into the next take's
first syllable (the tail is capped at the next speech onset).
"""

import os
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from .align import Decision, LOW, MATCH
from .assemble import _split_runs
from .text import normalize
from .transcribe import Word
from .util import Progress, fmt_ts

_EXPLAIN = set(filter(None, os.environ.get("AUDIO_CLEANER_EXPLAIN", "").split(",")))

LONG_WORD_S = 1.0          # a transcribed word this long is hiding other speech
GAP_MIN_S = 0.35           # inter-word gaps shorter than this are normal articulation
GAP_SPEECH_FRACTION = 0.4  # ...longer ones count only if this much of them is speech
PROBE_LEN_S = 8.0          # mid-span probes
BASE_LEN_S = 20.0          # a run's own hearing, from its start
PROBE_STEP_S = 0.5         # probe starts are staggered through a stretched word
WORD_MATCH = 75            # fuzzy score for a heard word to count as a script word
REPEAT_MIN_WORDS = 2       # a repeat must re-say at least this many consecutive script words
REPEAT_MIN_GAP_S = 0.6     # ...and land this long after the words were first heard
RESTART_MAX_INDEX = 2      # a repeat from this early in the sentence is a restart: drop all before it
RESTART_MIN_WORDS = 3      # ...if the new attempt carries on for at least this many words
OPENING_WORDS = 5          # a fuzzy re-hearing of this many opening words is also a restart
OPENING_SCORE = 75
OPENING_SCORE_SHORT = 72   # for the three-word check
KNOWN_S = 0.4              # a probe word this close to a base word is the same word, not new speech
RESTART_CLUSTER_S = 1.5    # sightings of the latest attempt spread over this much (timestamp jitter)
RESTART_SNAP_S = 0.7       # snap the boundary to a silence this close before the estimate
MIN_KEPT_FRACTION = 0.45   # never cut a span below this share of its expected spoken length
WORDS_PER_S = 3.5          # fast speech; used only for that sanity floor
LEAD_MAX_S = 0.8           # how far back a late first-word timestamp may be corrected
LEAD_SILENCE_S = 0.2       # ...but only to an onset preceded by real silence
LEAD_UNCONFIRMED_S = 0.5   # ...and, when the first word cannot be confirmed, only this far


class Energy:
    """20 ms frame loudness of a 16 kHz mono WAV, with a simple speech/silence mask."""

    SR = 16000
    HOP = 320  # 20 ms

    def __init__(self, wav_path: str):
        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == self.SR and wf.getnchannels() == 1, "expected 16 kHz mono"
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.pcm = pcm.astype(np.float32) / 32768.0
        n = len(self.pcm) // self.HOP
        rms = np.sqrt((self.pcm[: n * self.HOP].reshape(n, self.HOP) ** 2).mean(axis=1)) + 1e-9
        self.db = 20 * np.log10(rms)
        floor = float(np.percentile(self.db, 10))
        self.speech = self.db > max(floor + 15.0, -45.0)
        self.duration = len(self.pcm) / self.SR

    def _frame(self, t: float) -> int:
        return min(len(self.speech), max(0, int(t * self.SR / self.HOP)))

    def speech_fraction(self, a: float, b: float) -> float:
        i, j = self._frame(a), self._frame(b)
        return float(self.speech[i:j].mean()) if j > i else 0.0

    def onset_before(self, t: float, max_back: float, floor: float) -> float:
        """Start of the speech run containing t, if it begins within max_back and after a real
        silence; otherwise t. Never earlier than floor."""
        i = self._frame(t)
        limit = max(self._frame(floor), i - int(max_back * self.SR / self.HOP))
        quiet = int(LEAD_SILENCE_S * self.SR / self.HOP)
        # Walk back looking for a silence at least LEAD_SILENCE_S long; speech resumes after it.
        j = i
        while j - quiet >= limit:
            if not self.speech[j - quiet:j].any():
                return min(t, j * self.HOP / self.SR)
            j -= 1
        return t

    def next_onset_after(self, t: float, max_ahead: float) -> float:
        """Time speech next begins after a silence following t (capped at t + max_ahead)."""
        j, end = self._frame(t), self._frame(t + max_ahead)
        while j < end and self.speech[j]:
            j += 1
        while j < end and not self.speech[j]:
            j += 1
        return j * self.HOP / self.SR if j < end else t + max_ahead

    def slice(self, a: float, b: float) -> np.ndarray:
        return self.pcm[max(0, int(a * self.SR)):max(0, int(b * self.SR))]


@dataclass
class Heard:
    norm: str
    start: float  # absolute seconds
    end: float


class Prober:
    """Transcribes short stretches of audio in isolation (numpy in, words with absolute times out)."""

    def __init__(self, whisper, language: str):
        self.whisper = whisper
        self.language = language
        self.calls = 0

    def words(self, energy: Energy, a: float, b: float) -> list[Heard]:
        a, b = max(0.0, a), min(energy.duration, b)
        if b - a < 0.3:
            return []
        self.calls += 1
        segments, _ = self.whisper.transcribe(energy.slice(a, b), language=self.language,
                                              word_timestamps=True, beam_size=5)
        out = []
        for seg in segments:
            for w in seg.words or []:
                norm = normalize(w.word)
                if not norm:
                    continue
                h = Heard(norm, a + float(w.start), a + float(w.end))
                # When a window ends mid-sentence whisper pads it with echoed words that all carry
                # the same zero-length timestamp. They are not speech; drop them.
                if h.end - h.start < 0.03 or (out and h.start - out[-1].start < 0.03):
                    continue
                out.append(h)
        return out


def _mid_starts(span: list[Word], run: tuple[float, float], energy: Energy,
                max_gap_s: float) -> list[float]:
    """Where to listen again inside one run: through every stretched word, and at every gap that
    has speech in it."""
    a, b = run
    starts = []
    for i, w in enumerate(span):
        if not (a - 1e-6 <= w.start <= b + 1e-6):
            continue
        if w.end - w.start > LONG_WORD_S:
            t = w.start + 0.2
            while t < w.end - 0.3:
                starts.append(t)
                t += PROBE_STEP_S
        if i + 1 < len(span):
            nxt = span[i + 1]
            if GAP_MIN_S < nxt.start - w.end <= max_gap_s and \
                    energy.speech_fraction(w.end, nxt.start) > GAP_SPEECH_FRACTION:
                starts.append(w.end)
    return starts


def _opening_run(heard: list[Heard], tokens: list[str]):
    """(time, run) if this hearing begins with the sentence opening and carries on."""
    runs = _runs(_canonical(heard, tokens), tokens)
    if sum(n for _, _, n in runs) < min(5, len(tokens)):
        return None
    for j, i, n in runs[:3]:
        if i <= RESTART_MAX_INDEX and n >= min(2, len(tokens)):
            return heard[j].start
    return None


def _canonical(heard: list[Heard], tokens: list[str]) -> list[str | None]:
    """Map each heard word to the script word it most resembles (None if it resembles none)."""
    from rapidfuzz import fuzz, process
    vocab = list(dict.fromkeys(tokens))
    out = []
    for h in heard:
        hit = process.extractOne(h.norm, vocab, scorer=fuzz.ratio, score_cutoff=WORD_MATCH)
        out.append(hit[0] if hit else None)
    return out


def _runs(canon: list[str | None], tokens: list[str]) -> list[tuple[int, int, int]]:
    """Greedy runs of consecutive heard words matching consecutive script words, as
    (heard_index, script_index, length). Each heard word belongs to at most one run; among the
    script positions a run could start at, the longest run wins, then the one nearest the point the
    sentence had reached."""
    runs = []
    j, expected = 0, 0
    while j < len(canon):
        if canon[j] is None:
            j += 1
            continue
        best = None
        for i, tok in enumerate(tokens):
            if tok != canon[j]:
                continue
            length = 1
            while j + length < len(canon) and i + length < len(tokens):
                if canon[j + length] == tokens[i + length]:
                    length += 1
                elif j + length + 1 < len(canon) and i + length + 1 < len(tokens) \
                        and canon[j + length + 1] == tokens[i + length + 1]:
                    length += 2  # one word whisper garbled, run continues after it
                else:
                    break
            key = (length, -abs(i - expected))
            if best is None or key > best[0]:
                best = (key, i, length)
        _, i, length = best
        runs.append((j, i, length))
        expected = i + length
        j += length
    return runs


def _opening_again(heard: list[Heard], tokens: list[str]) -> list[float]:
    """Times at which this hearing says the unit's opening words a second (third...) time, judged
    by fuzzy ratio over a few words, so a garbled retake ("the mind was caught of Vladimir" for
    "That man was Colonel Vladimir") still counts."""
    from rapidfuzz import fuzz
    if len(tokens) < 3:
        return []
    again: list[float] = []
    # A longer opening is checked loosely, a three-word one strictly; either sighting counts.
    for n, cutoff in ((min(OPENING_WORDS, len(tokens)), OPENING_SCORE), (3, OPENING_SCORE_SHORT)):
        opening = " ".join(tokens[:n])
        # The same words further into the unit (not a one-word shift of the opening itself).
        later = [" ".join(tokens[i:i + n]) for i in range(3, len(tokens) - n + 1)]
        first = None
        for j in range(len(heard) - n + 1):
            window = " ".join(h.norm for h in heard[j:j + n])
            score = fuzz.ratio(opening, window)
            if score < cutoff:
                continue
            # "he used his" is not a retake of "who used it" when the unit itself says it later.
            if any(fuzz.ratio(l, window) >= score for l in later):
                continue
            t = heard[j].start
            if first is None:
                first = t
            elif t > first + REPEAT_MIN_GAP_S and all(abs(t - a) > REPEAT_MIN_GAP_S for a in again):
                again.append(t)
    return sorted(again)


def _blank(base: list[Heard], t: float) -> bool:
    """True when the span's own hearing has no word near t: speech there was hidden."""
    return not any(abs(h.start - t) < KNOWN_S for h in base)


def _repeats(heard: list[Heard], tokens: list[str]) -> list[tuple[float, float, int, int]]:
    """Places where the speaker went back and said script words again, as
    (first_said, said_again, script_index, run_length)."""
    canon = _canonical(heard, tokens)
    first_time_at: dict[int, float] = {}
    reached = 0
    found = []
    for j, i, length in _runs(canon, tokens):
        t = heard[j].start
        if length >= REPEAT_MIN_WORDS and i < reached and i in first_time_at \
                and t > first_time_at[i] + REPEAT_MIN_GAP_S:
            found.append((first_time_at[i], t, i, length))
        for k in range(length):
            first_time_at.setdefault(i + k, heard[j + k].start)
        reached = max(reached, i + length)
    return found


def _attempt_start(heard: list[Heard], tokens: list[str], t: float) -> float:
    """Walk back from the word at t over words whisper garbled or that sit before it in the
    script, as long as there is no real pause: that is where this attempt actually began."""
    canon = _canonical(heard, tokens)
    idx = next((k for k, h in enumerate(heard) if h.start >= t - 1e-6), None)
    if idx is None:
        return t
    i_here = tokens.index(canon[idx]) if canon[idx] in tokens else 0
    start = heard[idx].start
    k = idx - 1
    while k >= 0 and start - heard[k].end < 0.45:
        c = canon[k]
        if c is not None and tokens.index(c) >= i_here:
            break
        start = heard[k].start
        k -= 1
    return start


def _blank_raw(words: list[Word], t: float) -> bool:
    """True when the transcript has no word starting near t: whatever is said there was hidden
    (swallowed into a stretched word) and never seen by the aligner."""
    return not any(abs(w.start - t) < KNOWN_S for w in words)


def _matched_indices(heard: list[Heard], tokens: list[str]) -> list[tuple[int, int]]:
    """(heard_index, script_index) for heard words that are script words, assigned in order."""
    canon = _canonical(heard, tokens)
    out, expected = [], 0
    for j, c in enumerate(canon):
        if c is None:
            continue
        positions = [i for i, tok in enumerate(tokens) if tok == c]
        i = min(positions, key=lambda i: (abs(i - expected), i))
        out.append((j, i))
        expected = i + 1
    return out


def _hidden_attempt(heard: list[Heard], tokens: list[str]) -> float | None:
    """A probe that starts inside hidden speech and hears the unit from its beginning: three
    matched script words climbing from index <= 2, allowing a couple of stray matches first (a
    function word the unit also uses later). Returns where the attempt begins, or None when the
    probe evidently started inside the attempt."""
    m = _matched_indices(heard, tokens)
    for a in range(0, min(3, len(m) - 2)):
        (j1, i1), (_, i2), (_, i3) = m[a], m[a + 1], m[a + 2]
        if i1 <= RESTART_MAX_INDEX and i1 < i2 < i3 and i3 - i1 <= 6:
            break
    else:
        return None
    # Walk back over words glued to the first matched one (garbled first words, strays).
    k = j1
    while k > 0 and heard[k].start - heard[k - 1].end < 0.45:
        k -= 1
    if k == j1 and i1 > 0:
        return None  # began mid-attempt: the first word(s) lie before this probe's start
    return heard[k].start


def _first_word_at(heard: list[Heard], tokens: list[str]) -> float | None:
    """Time of the first hearing of the unit's first word that is followed, within three words,
    by another of the unit's first four words."""
    from rapidfuzz import fuzz
    for j, h in enumerate(heard):
        if fuzz.ratio(h.norm, tokens[0]) < 80:
            continue
        follow = [x.norm for x in heard[j + 1:j + 4]]
        if any(fuzz.ratio(a, b) >= 75 for a in follow for b in tokens[1:4]):
            return h.start
    return None


def _foreign_lead(heard: list[Heard], tokens: list[str], words: list[Word]) -> float | None:
    """The unit opens inside this hearing after two or more words that are not its own, said
    where the transcript was blank: the speaker restarted from an earlier clause. Returns where
    this unit's own words begin."""
    t = _opening_run(heard, tokens)
    if t is None:
        return None
    before = [h for h in heard if h.start < t - 0.05]
    if len(before) < 2:
        return None
    canon = _canonical(before, tokens)
    if any(c is not None for c in canon[-2:]):
        return None
    if not all(_blank_raw(words, h.start) for h in before[-2:]):
        return None
    return t


def _pick_restart(exact: list[float], fuzzy: list[float], energy: Energy, floor: float) -> float | None:
    """Where the kept attempt begins, from all sightings of restarts.

    Sightings of one attempt differ by whisper's timestamp jitter (over a second inside stretched
    words), so the latest attempt is taken as everything within RESTART_CLUSTER_S of the latest
    sighting, exact word runs are trusted over fuzzy openings, and the boundary is then snapped
    to the silence before the attempt when there is one.
    """
    hits = exact + fuzzy
    if not hits:
        return None
    t_max = max(hits)
    cluster_exact = sorted(t for t in exact if t >= t_max - RESTART_CLUSTER_S)
    cluster_fuzzy = sorted(t for t in fuzzy if t >= t_max - RESTART_CLUSTER_S)
    # Probes that start before the attempt report it early, probes that start inside it report
    # it late; the median of the sightings sits close to the truth.
    pool = cluster_exact or cluster_fuzzy
    t_est = pool[len(pool) // 2]
    low = max(floor, min(cluster_exact + cluster_fuzzy) - 0.3)
    onset = energy.onset_before(t_est + 0.15, RESTART_SNAP_S, floor=low)
    if onset < t_est + 0.15:
        return onset - 0.02
    return t_est - 0.08


def refine(decisions: list[Decision], words: list[Word], energy: Energy, prober: Prober,
           max_gap_s: float) -> int:
    """Adjust kept spans in place: drop hidden earlier attempts, cut stutters, fix a late first-word
    timestamp. Returns the number of spans changed."""
    kept = [d for d in decisions if d.status in (MATCH, LOW)]
    progress = Progress("Refine")
    changed = 0
    prev_end = 0.0
    for k, d in enumerate(kept):
        span = words[d.word_start:d.word_end]
        tokens = d.sentence.tokens
        runs_t = _split_runs(words, d.word_start, d.word_end, max_gap_s)
        near = [w for w in words if d.start - 3.0 <= w.start <= d.end + 3.0]

        notes = []
        new_start = d.start
        excise = []
        repeats = []          # (first, again, idx, n) seen inside one probe
        exact, fuzzy = [], []  # restart sightings
        for r, (ra, rb) in enumerate(runs_t):
            base = prober.words(energy, ra - 0.25, min(rb + 0.3, ra + BASE_LEN_S))
            probes = [prober.words(energy, t, min(rb + 0.3, t + PROBE_LEN_S))
                      for t in _mid_starts(span, (ra, rb), energy, max_gap_s)]
            for p in [base] + probes:
                if not p:
                    continue
                for first, again, i, n in _repeats(p, tokens):
                    if again >= d.end - 0.3:
                        continue
                    if i <= RESTART_MAX_INDEX and n >= min(RESTART_MIN_WORDS, len(tokens)):
                        exact.append(_attempt_start(p, tokens, again))
                    else:
                        repeats.append((first, again, i, n))
                fuzzy += [t for t in _opening_again(p, tokens) if t < d.end - 0.3]
                t = _foreign_lead(p, tokens, near)
                if t is not None and t < d.end - 0.3:
                    exact.append(t)
            # The run's own hearing starts with words that are not the unit's, then the unit's
            # first word: a fragment of a discarded attempt glued to the front (early timestamp).
            if r == 0 and base:
                t_base = _first_word_at(base, tokens)
                if t_base is not None:
                    before = [h for h in base if h.start < t_base - 0.05]
                    foreign = len(before) >= 2 and \
                        sum(c is not None for c in _canonical(before[-2:], tokens[:4])) <= 1
                    sightings = []
                    for q, q0 in [(base, ra - 0.25)] + list(zip(probes, _mid_starts(span, (ra, rb), energy, max_gap_s))):
                        tq = _first_word_at(q, tokens)
                        if tq is not None and tq - q0 >= 0.4:
                            sightings.append(tq)
                    sightings.sort()
                    t = sightings[len(sightings) // 2] if sightings else t_base
                    if foreign and t > d.start + 0.3:
                        exact.append(t)
            # A later run that opens the unit again is a fresh attempt after a long pause.
            if r > 0 and base:
                t = _opening_run(base, tokens)
                if t is not None:
                    exact.append(t)
            # A probe that hears the unit's beginning inside speech the transcript never saw.
            for p in probes:
                if not p or not _blank_raw(near, p[0].start):
                    continue
                t = _hidden_attempt(p, tokens)
                if t is not None and d.start + 0.3 < t < d.end - 0.3:
                    exact.append(t)

        picked = _pick_restart(exact, fuzzy, energy, floor=prev_end + 0.05)
        if str(d.sentence.index) in _EXPLAIN:
            print(f"\n[explain #{d.sentence.index}] span {d.start:.2f}-{d.end:.2f} runs={[(round(a, 2), round(b, 2)) for a, b in runs_t]}")
            print(f"  exact={[round(t, 2) for t in exact]} fuzzy={[round(t, 2) for t in fuzzy]} repeats={[(round(a, 2), round(b, 2), i, n) for a, b, i, n in repeats]} picked={picked}")
            for r, (ra, rb) in enumerate(runs_t):
                base = prober.words(energy, ra - 0.25, min(rb + 0.3, ra + BASE_LEN_S))
                print(f"  run{r} base: " + " ".join(f"{h.norm}@{h.start:.2f}" for h in base))
                for t0 in _mid_starts(span, (ra, rb), energy, max_gap_s):
                    p0 = prober.words(energy, t0, min(rb + 0.3, t0 + PROBE_LEN_S))
                    print(f"  mid@{t0:.2f}: " + " ".join(f"{h.norm}@{h.start:.2f}" for h in p0)[:220])
        if picked is not None and picked > d.start + 0.3:
            new_start = picked
            notes.append(f"restart: kept attempt from {fmt_ts(new_start)}")
        for first, again, i, n in repeats:
            if i > RESTART_MAX_INDEX and first >= new_start and 0.5 <= again - first <= 6.0:
                excise.append((first, again - 0.03))
        excise = _merge([(a, b) for a, b in excise if b - a >= 0.25 and a >= new_start])
        if excise:
            notes.append("re-said: cut " + ", ".join(f"{fmt_ts(a)}-{fmt_ts(b)}" for a, b in excise))

        # A late first-word timestamp clips the unit's start; extend back to the real onset.
        if new_start == d.start:
            onset = energy.onset_before(new_start, LEAD_MAX_S, floor=prev_end + 0.05)
            if onset < new_start - 0.08:
                # Only if the stretch holds the unit's own first word, not a fragment of a
                # discarded attempt glued to it.
                p = prober.words(energy, onset - 0.05, new_start + 1.5)
                t_open = _first_word_at(p, tokens)
                if t_open is None:
                    # Whisper garbled the opening; trust a short extension, not a long one that
                    # could hold a fragment of a discarded attempt.
                    if new_start - onset > LEAD_UNCONFIRMED_S:
                        onset = new_start
                elif t_open > onset + 0.15:
                    onset = max(onset, t_open - 0.04)
                if onset < new_start - 0.08:
                    notes.append(f"lead +{new_start - onset:.2f}s")
                    new_start = onset

        # Sanity floor: a span can lose retakes, not most of its words.
        kept_s = (d.end - new_start) - sum(b - a for a, b in excise)
        if notes and kept_s < MIN_KEPT_FRACTION * len(tokens) / WORDS_PER_S:
            d.notes = [f"left alone: would keep only {kept_s:.1f}s ({'; '.join(notes)})"]
            prev_end = d.end
            progress.update(f"{k + 1}/{len(kept)} spans — {changed} adjusted, {prober.calls} probes")
            continue

        if notes:
            d.start, d.excise, d.notes = new_start, excise, notes
            changed += 1
        prev_end = d.end
        progress.update(f"{k + 1}/{len(kept)} spans — {changed} adjusted, {prober.calls} probes")
    progress.done(f"done — {changed}/{len(kept)} spans adjusted ({prober.calls} probes)")
    return changed


def _merge(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
