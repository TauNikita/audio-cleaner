"""Cut the chosen spans and concatenate them with pauses into the clean output.

Every piece of audio is cut at its *audible* edges, found on the 16 kHz energy track against the
recording's noise floor, not at whisper's word timestamps: a word's timestamp ends a median 180 ms
before its sound does (and sometimes half a second after it), a first word's timestamp lands early
by the VAD's padding (400-800 ms) or late inside the word, and a "word" of 10 ms at a segment start
has no sound at all. So:

- the tail runs from the last word's timestamp to where the sentence's sound dies down for good: a
  sustained quiet, longer than the closure before a final stop or the dip before a trailing "s" or
  "f" (those stay under ~120 ms), and keeps a fixed margin of room after it, so the decay is never
  clipped and the fade lands in near-silence; a timestamp that is already in the quiet after the
  word is walked back to the sound;
- the lead is a fixed pre-roll of room before the first sound that looks like speech (a breath or
  a click a few dB over the floor does not): an early timestamp's dead air is trimmed, a timestamp
  inside the first sound moves to that sound's start when it is a clean onset, and otherwise
  (speech glued to earlier speech) the cut falls at the quietest moment, as before;
- with both margins fixed, the silence inserted between two pieces makes the pause the listener
  hears — last sound to first sound — exactly the configured length, whatever room the takes had.

A silence inside a span longer than --max-inner-pause (the speaker finding the line) is shortened to
the clause pause; a piece whose timestamps hold no sound (a transcript artifact) is dropped. The
stretches hidden.refine() excised (retakes inside a sentence) and take switches inside running
speech are hard joins: the cut falls at the quietest moment next to the word boundary and gets a
longer fade. Everything is written to <output>.cuts.json so the joins can be audited.
"""

import json
import os
import re
import sys

import numpy as np

from .align import Decision, LOW, MATCH
from .transcribe import Word
from .util import Progress, fmt_ts

# A transcript word this long is whisper covering speech it did not write out, and a short gap
# right after it is where the lost words were said. A long gap after it is a pause or a retake.
STRETCHED_WORD_S = 1.0
STRETCHED_GAP_MAX_S = 2.5

# Audible edges, on 10 ms frames of the 16 kHz mono track.
FINE_HOP_MS = 10
EDGE_ABOVE_FLOOR_DB = 10.0   # a frame this far above the noise floor (5th percentile) is sound
BLIP_MS = 40                 # loud runs shorter than this are clicks and noise, not speech
SPEECH_BELOW_PEAK_DB = 15.0  # a run of sound whose peak stays further below the piece's speech level
                             # is a breath or a noise, not a word (breaths sit 20-26 dB below)...
SOFT_WORD_BELOW_PEAK_DB = 24.0  # ...unless it lasts SOFT_WORD_MIN_MS and comes within this: a soft
SOFT_WORD_MIN_MS = 100          # first word ("In", "a") spoken under the sentence's level
SPEECH_MIN_MS = 60           # a word's sound lasts at least this, or more sound follows within
SPEECH_FOLLOW_MS = 150       # this (a short initial burst)
TAIL_SETTLE_MS = 180         # quiet this long after a word ends the sentence: longer than the closure
                             # before a final stop or the dip before a weak fricative (<= ~120 ms)
TAIL_SETTLE_NEXT_MS = 130    # ...or this long when speech of at least NEW_SOUND_MS follows: the
NEW_SOUND_MS = 150           # sentence did end, the next take began
TAIL_BACK_MS = 1500          # how far back a last-word timestamp that lands in the quiet may be
                             # walked to the sound it belongs to (the VAD's padding runs long)
ONSET_FORWARD_MS = 1200      # how far past an early first-word timestamp the first speech is looked for
ONSET_BACK_MS = 700          # how far back a timestamp inside the first word may move to its start
ONSET_CLEAR_MS = 150         # ...when that much quiet precedes the sound (a clean onset, not glued speech)
SILENT_PIECE_LOOK_MS = 500   # a piece with no speech from 100 ms before its timestamps to this far
                             # after them is a transcript artifact and is dropped
FADE_IN_MS = 30              # fades land in the kept room; a cut that has to fall in sound gets hard_fade
FADE_OUT_MS = 60
DIP_FADE_MS = 12             # a cut in a quiet dip glued to the word gets only an anti-click fade


def _drop_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """True runs shorter than min_run frames become False."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i < min_run:
            out[i:j] = False
        i = j
    return out


class _Fine:
    """10 ms loudness of the 16 kHz mono track, with a sound mask against the noise floor and a
    speech mask (sound at the level of the piece's own words) for the edge finders."""

    SR = 16000
    HOP = SR * FINE_HOP_MS // 1000

    def __init__(self, pcm: np.ndarray):
        n = len(pcm) // self.HOP
        rms = np.sqrt((pcm[:n * self.HOP].reshape(n, self.HOP) ** 2).mean(axis=1))
        self.db = 20 * np.log10(rms + 1e-7)
        self.floor = float(np.percentile(self.db, 5))
        self.level = float(np.percentile(self.db, 95))    # the recording's speech level
        self.loud = _drop_short_runs(self.db > self.floor + EDGE_ABOVE_FLOOR_DB, BLIP_MS // FINE_HOP_MS)
        self.n = n

    def f(self, t: float) -> int:
        return min(self.n, max(0, int(round(t * 1000 / FINE_HOP_MS))))

    def t(self, f: int) -> float:
        return f * FINE_HOP_MS / 1000.0

    def speech_ref(self, a: float, b: float) -> float:
        """The speech level of a stretch: the 95th percentile of its loud frames, or the recording's
        when the stretch holds too little sound to say."""
        i, j = self.f(a), self.f(b)
        seg = self.db[i:j][self.loud[i:j]] if j > i else np.array([])
        return float(np.percentile(seg, 95)) if len(seg) >= 20 else self.level

    def speech(self, a: float, b: float, ref: float) -> tuple[int, np.ndarray]:
        """The sound mask over [a, b) with runs that cannot be words (peak more than
        SPEECH_BELOW_PEAK_DB under ref, or too short and alone) set quiet. Returns (first frame, mask)."""
        i, j = self.f(a), self.f(b)
        m = self.loud[i:j].copy()
        k, n = 0, len(m)
        follow = SPEECH_FOLLOW_MS // FINE_HOP_MS
        while k < n:
            if not m[k]:
                k += 1
                continue
            h = k
            while h < n and m[h]:
                h += 1
            peak = float(self.db[i + k:i + h].max())
            length_ms = (h - k) * FINE_HOP_MS
            short = length_ms < SPEECH_MIN_MS and not self.loud[i + h:i + h + follow].any()
            soft_word = length_ms >= SOFT_WORD_MIN_MS and peak >= ref - SOFT_WORD_BELOW_PEAK_DB
            if short or (peak < ref - SPEECH_BELOW_PEAK_DB and not soft_word):
                m[k:h] = False
            k = h
        return i, m

    def speech_in(self, a: float, b: float, ref: float) -> bool:
        _, m = self.speech(a, b, ref)
        return bool(m.any())

    def sound_in(self, a: float, b: float) -> bool:
        i, j = self.f(a), self.f(b)
        return bool(self.loud[i:j].any()) if j > i else False

    def quietest(self, a: float, b: float) -> float:
        j = min(self.n, max(self.f(a) + 1, self.f(b)))
        i = max(0, min(self.f(a), j - 1))
        return self.t(i + int(np.argmin(self.db[i:j])))

    def audible_end(self, t_from: float, t_limit: float, ref: float) -> tuple[float, bool]:
        """Where the speech running at t_from dies down for good: the start of the first quiet lasting
        TAIL_SETTLE_MS (or TAIL_SETTLE_NEXT_MS and then NEW_SOUND_MS of speech), looking from a little
        before t_from (a slightly late timestamp) up to t_limit. A timestamp that already sits in the
        quiet is walked back (up to TAIL_BACK_MS) to the speech it belongs to. Returns (end, settled);
        (t_limit, False) when the speech never settles before the limit."""
        settle = TAIL_SETTLE_MS // FINE_HOP_MS
        settle_next = TAIL_SETTLE_NEXT_MS // FINE_HOP_MS
        new = NEW_SOUND_MS // FINE_HOP_MS
        back = TAIL_BACK_MS // FINE_HOP_MS
        a = max(0.0, t_from - TAIL_BACK_MS / 1000.0 - 0.1)
        i0, m = self.speech(a, t_limit + (TAIL_SETTLE_MS + 100) / 1000.0, ref)
        n = len(m)
        f0 = self.f(t_from) - i0
        lim = self.f(t_limit) - i0

        def run_from(g: int) -> int:
            h = g
            while h < n and m[h]:
                h += 1
            return h - g

        # The timestamp lands in the quiet: the word ended earlier than whisper says. Walk back to
        # the speech, and take its end if the quiet after it is a real pause (long, or followed by
        # speech after a shorter one), not a dip inside the last word.
        if 0 <= f0 < n and not m[max(0, f0 - 5):f0 + 1].any():
            k = f0
            while k > 0 and f0 - k < back and not m[k - 1]:
                k -= 1
            if k > 0 and f0 - k < back:
                g = k
                while g < n and not m[g]:
                    g += 1
                qlen = g - k
                if qlen >= settle or (qlen >= settle_next and (g >= n or run_from(g) >= new)):
                    return self.t(i0 + k), True
        g = max(0, f0 - 5)
        while g < lim:
            if m[g]:
                g += 1
                continue
            qa = g
            stop = min(n, lim + settle)
            while g < stop and not m[g]:
                g += 1
            qlen = g - qa
            if qlen >= settle or (qlen >= settle_next and (g >= n or run_from(g) >= new)):
                return max(self.t(i0 + qa), t_from - 0.05), True
        return t_limit, False

    def audible_onset(self, t_guess: float, lo: float, ref: float) -> float | None:
        """Where the speech at/after a timestamp begins: the first speech within ONSET_FORWARD_MS of
        an early timestamp, or the start of the speech a late timestamp sits in when that is a clean
        onset (quiet before it, within ONSET_BACK_MS, not before lo). t_guess itself when no speech
        follows within the look-ahead. None when the timestamp sits in speech glued to earlier speech
        (no clean onset to move to)."""
        a = max(0.0, min(lo, t_guess - ONSET_BACK_MS / 1000.0) - 0.2)
        i0, m = self.speech(a, t_guess + ONSET_FORWARD_MS / 1000.0 + 0.1, ref)
        n = len(m)
        f0 = min(n - 1, self.f(t_guess) - i0)
        if f0 < 0 or n == 0:
            return t_guess
        if not m[f0]:
            g, lim = f0, min(n, f0 + ONSET_FORWARD_MS // FINE_HOP_MS)
            while g < lim and not m[g]:
                g += 1
            return self.t(i0 + g) if g < lim else t_guess
        g, lim = f0, max(self.f(lo) - i0, f0 - ONSET_BACK_MS // FINE_HOP_MS, 0)
        while g > lim and m[g - 1]:
            g -= 1
        clear = ONSET_CLEAR_MS // FINE_HOP_MS
        if (g > 0 and m[g - 1]) or m[max(0, g - clear):g].any():
            return None
        return self.t(i0 + g)

    def quiet_stretches(self, a: float, b: float, min_s: float) -> list[tuple[float, float]]:
        """Quiet runs of at least min_s that end inside (a, b), as (start, end) seconds; a quiet
        already running at a is reported from a (the caller trims it: dead air before an early
        first-word timestamp)."""
        i, j = self.f(a), self.f(b) - 1
        out = []
        g = i
        while g < j:
            if self.loud[g]:
                g += 1
                continue
            qa = g
            while g < j and not self.loud[g]:
                g += 1
            if g < j and self.t(g) - self.t(qa) >= min_s:
                out.append((self.t(qa), self.t(g)))
        return out

    def speech_run_end(self, t: float, ref: float, max_ahead: float = 1.0) -> float:
        """Where the speech sounding at t stops (t itself when t is quiet), within max_ahead."""
        i0, m = self.speech(t, t + max_ahead, ref)
        g = 0
        while g < len(m) and m[g]:
            g += 1
        return self.t(i0 + g)


def _split_runs(words: list[Word], w_start: int, w_end: int,
                max_gap_s: float) -> list[tuple[float, float]]:
    """Split a kept word span into temporally-contiguous runs of (start, end) seconds.

    Whisper's chosen words are contiguous by index but not always by time: when vad_filter drops a
    flubbed retake, a few stray words straddle it with a large time gap. Cutting [first.start,
    last.end] as one block would re-include that retake / dead audio — the source of output repeats —
    so we break the span wherever the gap between consecutive words exceeds max_gap_s and drop the
    gap. A gap right after a word with a stretched timestamp is not split: whisper collapsed a
    stretch of speech into that word (a repeated phrase the script itself contains, say), and the
    gap is where the lost words were said; the hidden-retake pass probes stretched words anyway.
    """
    runs: list[tuple[float, float]] = []
    run_start = words[w_start].start
    prev_end = words[w_start].end
    prev_long = words[w_start].end - words[w_start].start > STRETCHED_WORD_S
    for wi in range(w_start + 1, w_end):
        w = words[wi]
        if w.start - prev_end > max_gap_s and not (prev_long and w.start - prev_end <= STRETCHED_GAP_MAX_S):
            runs.append((run_start, prev_end))
            run_start = w.start
        prev_end = w.end
        prev_long = w.end - w.start > STRETCHED_WORD_S
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


def _clip_runs(runs: list[tuple[float, float]], start: float, end: float,
               excise: list[tuple[float, float]], locked: bool = False) -> list[tuple[float, float, bool, bool]]:
    """Trim runs to [start, end] and cut the excised intervals out of them (see hidden.refine), as
    (start, end, hard_start, hard_end): an edge made by an excision is hard — the speech may run
    straight through it, so the cut there cannot wait for a silence. The inner edges of runs placed
    by hand (--overrides) are hard too: the stretches are the author's, what lies between them is not."""
    out: list[tuple[float, float, bool, bool]] = []
    for r, (a, b) in enumerate(runs):
        # The first run begins where refine() says the take begins: later than the first word's
        # timestamp after a restart, earlier than it when that timestamp landed late.
        pieces = [(start if r == 0 else max(a, start), min(b, end),
                   locked and r > 0, locked and r < len(runs) - 1)]
        for x, y in excise:
            nxt = []
            for pa, pb, hs, he in pieces:
                if y <= pa or x >= pb:
                    nxt.append((pa, pb, hs, he))
                else:
                    if x > pa:
                        nxt.append((pa, x, hs, True))
                    if y < pb:
                        nxt.append((y, pb, True, he))
            pieces = nxt
        out += [p for p in pieces if p[1] - p[0] > 0.05]
    return out


def _split_inner(pieces: list[tuple[float, float, bool, bool]], fine: _Fine,
                 min_s: float) -> tuple[list[tuple[float, float, bool, bool]], int]:
    """Break every piece at the silences inside it that last at least min_s (the speaker pausing to
    find the line); each side keeps its own edges and the join gets the clause pause."""
    if min_s <= 0:
        return pieces, 0
    out, cuts = [], 0
    for pa, pb, hs, he in pieces:
        cur_a, cur_hs = pa, hs
        for qa, qb in fine.quiet_stretches(pa, pb, min_s):
            if qa <= cur_a + 0.05:
                # Dead air at the piece's own start (a restart or first-word timestamp placed in the
                # silence before the words), or a stub of a few ms before it (the tail of the previous
                # unit's last word, when a start was clamped to that unit's end): the piece simply
                # begins where its own sound does.
                cur_a = qb
                continue
            out.append((cur_a, qa, cur_hs, False))
            cur_a, cur_hs = qb, False
            cuts += 1
        if pb - cur_a > 0.05:
            out.append((cur_a, pb, cur_hs, he))
    return out, cuts


def _drop_silent(pieces: list[tuple[float, float, bool, bool]], fine: _Fine
                 ) -> tuple[list[tuple[float, float, bool, bool]], list[tuple[float, float]]]:
    """Drop pieces whose timestamps hold no sound at all (whisper pins a word to a segment start it
    did not happen at); a unit keeps its pieces when every one of them would go."""
    kept = [p for p in pieces if fine.sound_in(p[0] - 0.1, p[1] + SILENT_PIECE_LOOK_MS / 1000.0)]
    if not kept:
        return pieces, []
    return kept, [(p[0], p[1]) for p in pieces if p not in kept]


def _edges(fine: _Fine, pa: float, pb: float, hard_start: bool, hard_end: bool, prev_cut_end: float,
           next_src_start: float | None, lead_s: float, tail_s: float, tail_max_s: float,
           hard_fade_ms: int) -> dict:
    """Where to cut one piece: the lead pre-roll before its first speech, the tail margin after its
    last speech, and the fades. prev_cut_end is where the previous piece's cut ended in the source
    (never reach back into it); next_src_start is where the next piece's words begin (never run on
    into them)."""
    ref = fine.speech_ref(pa, pb)
    # Lead: a fixed pre-roll of room before the first speech. A timestamp that falls inside the
    # previous piece's cut (a rescued or restarted unit placed too early) starts after that cut, so
    # no audio is ever heard twice; an edge made by an excision never reaches back into it.
    after_prev = 0 <= pa - prev_cut_end < 1.5
    if prev_cut_end - 1.5 < pa <= prev_cut_end:
        pa = prev_cut_end + 0.01
        after_prev = True
    floor_t = max(prev_cut_end + 0.01 if after_prev else 0.0, pa - 0.02 if hard_start else 0.0)
    lo = max(pa - ONSET_BACK_MS / 1000.0, floor_t)
    on = fine.audible_onset(pa, lo, ref)
    if on is None:
        # The timestamp sits in speech glued to earlier speech: cut at the quietest moment around it.
        cut_start = max(fine.quietest(max(floor_t, pa - lead_s), pa + 0.03), floor_t)
        on, lead_room, fade_in, clean_lead = pa, 0.0, hard_fade_ms, False
    else:
        cut_start = max(on - lead_s, floor_t)
        clean_lead = True
        if fine.speech_in(cut_start, on - 0.01, ref):
            # Earlier speech inside the pre-roll (the previous take's last word): cut at the dip.
            # A breath or a click there is room and stays.
            cut_start = fine.quietest(cut_start, on)
            clean_lead = not fine.loud[min(fine.n - 1, fine.f(cut_start))]
        lead_room = max(0.0, on - cut_start)
        # A short fade when the cut sits in a quiet dip right before the word, so the word's
        # onset is not softened; the long one only when the cut has to fall in sound.
        fade_in = (FADE_IN_MS if lead_room * 1000 >= FADE_IN_MS
                   else DIP_FADE_MS if clean_lead else hard_fade_ms)

    # Tail: run on to where the speech dies down for good, then keep a fixed margin of room.
    limit = pb + tail_max_s
    if hard_end:
        limit = min(limit, pb + 0.05)
    if next_src_start is not None and 0 <= next_src_start - pb < 1.5:
        # Never run on into the next piece's words — but its timestamp is only a timestamp (it may
        # sit inside this piece's last word): the cap is where the next piece's first speech is
        # heard to begin; only when that cannot be told (speech glued across the boundary) the
        # timestamp itself bounds the tail and the cut falls at the dip.
        cap = next_src_start - 0.02
        nxt = fine.audible_onset(next_src_start, max(pb, next_src_start - ONSET_BACK_MS / 1000.0), ref)
        if nxt is not None and nxt > pb + 0.05:
            cap = max(cap, nxt - 0.03)
        limit = min(limit, cap)
    limit = max(limit, pb)
    end, settled = fine.audible_end(pb, limit, ref)
    end = max(end, cut_start + 0.05)
    if settled:
        cut_end = min(limit, end + tail_s)
        cut_end = max(cut_end, cut_start + 0.05)
        tail_room = max(0.0, cut_end - end)
        fade_out = FADE_OUT_MS if tail_room * 1000 >= FADE_OUT_MS else hard_fade_ms
        clean_tail = tail_room * 1000 >= FADE_OUT_MS
    else:
        # The speech runs on past the limit (a hard join, a take switch inside running speech): cut
        # at the quietest moment next to the word boundary.
        cut_end = fine.quietest(max(cut_start + 0.05, pb - 0.15), max(pb - 0.14, min(limit, pb + 0.30)))
        cut_end = max(cut_end, cut_start + 0.05)
        end, tail_room, fade_out, clean_tail = cut_end, 0.0, hard_fade_ms, False
    return dict(cut_start=cut_start, cut_end=cut_end, onset=on, audible_end=end,
                lead_room_ms=round(lead_room * 1000), tail_room_ms=round(tail_room * 1000),
                fade_in_ms=fade_in, fade_out_ms=fade_out, settled=settled,
                clean_lead=clean_lead, clean_tail=clean_tail)


def cuts_log_path(output: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", output) + ".cuts.json"


def assemble(media: str, decisions: list[Decision], words: list[Word], output: str, energy, *,
             lead_ms: int, tail_ms: int, tail_max_ms: int, sentence_pause_ms: int,
             paragraph_pause_ms: int, clause_pause_ms: int, max_internal_gap_ms: int,
             max_inner_pause_ms: int, hard_fade_ms: int) -> list[tuple[Decision, float, float]]:
    """Render the kept spans to `output` (format from the extension) and write <output>.cuts.json.
    Returns where each kept sentence landed in the output as (decision, start_s, end_s), for verify().

    Spans are cut from the original recording, not the 16 kHz mono WAV used for transcription, so the
    output keeps the source sample rate and channels; the edges are found on the 16 kHz track
    (`energy`, hidden.Energy of the same recording) and applied to the original by time. A pause
    value is the silence the listener hears between two pieces, from the last sound of one to the
    first sound of the next: the kept room (lead_ms before an onset, tail_ms after an end) counts
    towards it and only the remainder is inserted as silence.
    """
    AudioSegment = _load_pydub()
    fine = _Fine(energy.pcm)
    max_gap_s = max_internal_gap_ms / 1000.0
    audio = AudioSegment.from_file(media)
    frame_rate, channels = audio.frame_rate, audio.channels
    total_ms = len(audio)

    kept = [d for d in decisions if d.status in (MATCH, LOW)]
    if not kept:
        raise RuntimeError("nothing to render: no sentences matched")

    # Every piece of every unit, in output order, with the unit it belongs to.
    plan: list[tuple[int, int, tuple[float, float, bool, bool]]] = []
    inner_cuts = 0
    dropped: list[dict] = []
    no_audio: list[int] = []
    for i, d in enumerate(kept):
        pieces = _clip_runs(d.runs or _split_runs(words, d.word_start, d.word_end, max_gap_s),
                            d.start, d.end, d.excise, locked=d.locked)
        pieces, n_inner = _split_inner(pieces, fine, max_inner_pause_ms / 1000.0)
        pieces, gone = _drop_silent(pieces, fine)
        inner_cuts += n_inner
        dropped += [dict(unit=d.sentence.index, src_start=round(a, 3), src_end=round(b, 3)) for a, b in gone]
        if not pieces and d.end - d.start > 0.05:
            pieces = [(d.start, d.end, False, False)]   # whatever the span holds, rather than nothing
        if not pieces:
            print(f"warning: sentence #{d.sentence.index} yields no audio "
                  f"({fmt_ts(d.start)}-{fmt_ts(d.end)}); it is missing from the output", file=sys.stderr)
            no_audio.append(d.sentence.index)
            continue
        plan += [(i, j, p) for j, p in enumerate(pieces)]

    progress = Progress("Assemble")
    result = AudioSegment.empty()
    positions: list[tuple[Decision, float, float]] = []
    log_units: list[dict] = []
    prev_cut_end = -10.0
    prev_tail_room_ms = 0
    hard_joins = in_sound = 0
    unit_entry = None
    skipped: set[int] = set()
    for k, (i, j, (pa, pb, hard_start, hard_end)) in enumerate(plan):
        d = kept[i]
        if 0 <= prev_cut_end - pb + 0.05 < 1.5 and pb <= prev_cut_end + 0.05:
            # The whole piece lies inside the previous piece's cut (a start placed on the previous
            # unit's last word): nothing of it is new sound.
            dropped.append(dict(unit=d.sentence.index, src_start=round(pa, 3), src_end=round(pb, 3),
                                reason="inside the previous piece's cut"))
            skipped.add(k)
            continue
        if j > 0 and all(kk in skipped for kk in range(k - j, k)):
            j = 0   # every earlier piece of this unit was dropped: this one opens the unit
        next_src = plan[k + 1][2][0] if k + 1 < len(plan) else None
        e = _edges(fine, pa, pb, hard_start, hard_end, prev_cut_end, next_src,
                   lead_ms / 1000.0, tail_ms / 1000.0, tail_max_ms / 1000.0, hard_fade_ms)
        if hard_start or hard_end:
            hard_joins += 1
        if not e["clean_lead"] or not e["clean_tail"]:
            in_sound += 1

        start_ms = max(0, int(round(e["cut_start"] * 1000)))
        end_ms = min(total_ms, int(round(e["cut_end"] * 1000)))
        sub = audio[start_ms:end_ms]
        fi, fo = min(e["fade_in_ms"], len(sub) // 2), min(e["fade_out_ms"], len(sub) // 2)
        if fi > 0:
            sub = sub.fade_in(fi)
        if fo > 0:
            sub = sub.fade_out(fo)

        # The pause heard between this piece and the previous one: between units by the kind of
        # boundary, inside a unit the clause pause. The room both pieces kept is part of it.
        pause_ms = 0
        if k > 0:
            if j > 0:
                pause_ms = clause_pause_ms
            else:
                prev = kept[i - 1].sentence
                pause_ms = (paragraph_pause_ms if prev.paragraph_end
                            else sentence_pause_ms if prev.sentence_end else clause_pause_ms)
        silence_ms = max(0, pause_ms - prev_tail_room_ms - e["lead_room_ms"]) if k > 0 else 0
        if silence_ms > 0:
            result += AudioSegment.silent(duration=silence_ms, frame_rate=frame_rate).set_channels(channels)

        out_start = len(result) / 1000.0
        result += sub
        out_end = len(result) / 1000.0
        piece_log = dict(unit=d.sentence.index, piece=j, src_start=round(pa, 3), src_end=round(pb, 3),
                         cut_start=round(e["cut_start"], 3), cut_end=round(e["cut_end"], 3),
                         onset=round(e["onset"], 3), audible_end=round(e["audible_end"], 3),
                         lead_room_ms=e["lead_room_ms"], tail_room_ms=e["tail_room_ms"],
                         fade_in_ms=fi, fade_out_ms=fo, settled=e["settled"],
                         clean_lead=e["clean_lead"], clean_tail=e["clean_tail"],
                         hard_start=hard_start, hard_end=hard_end, pause_ms=pause_ms,
                         silence_before_ms=silence_ms, out_start=round(out_start, 3), out_end=round(out_end, 3))
        if j == 0:
            unit_entry = dict(unit=d.sentence.index, status=d.status, score=round(d.score, 1),
                              text=d.sentence.raw, paragraph_end=d.sentence.paragraph_end,
                              out_start=round(out_start, 3), out_end=round(out_end, 3), pieces=[])
            log_units.append(unit_entry)
        unit_entry["pieces"].append(piece_log)
        unit_entry["out_end"] = round(out_end, 3)
        prev_cut_end, prev_tail_room_ms = e["cut_end"], e["tail_room_ms"]
        progress.update(f"{k + 1}/{len(plan)} pieces — output {fmt_ts(len(result) / 1000)}")
    by_index = {d.sentence.index: d for d in kept}
    for u in log_units:
        positions.append((by_index[u["unit"]], u["out_start"], u["out_end"]))

    ext = os.path.splitext(output)[1].lstrip(".").lower() or "wav"
    progress.update(f"encoding {output} ({ext})", force=True)
    result.export(output, format=ext)
    log = dict(media=os.path.abspath(media), output=os.path.abspath(output),
               duration_s=round(len(result) / 1000.0, 3), noise_floor_dbfs=round(fine.floor, 1),
               speech_level_dbfs=round(fine.level, 1),
               settings=dict(lead_ms=lead_ms, tail_ms=tail_ms, tail_max_ms=tail_max_ms,
                             sentence_pause_ms=sentence_pause_ms, paragraph_pause_ms=paragraph_pause_ms,
                             clause_pause_ms=clause_pause_ms, max_internal_gap_ms=max_internal_gap_ms,
                             max_inner_pause_ms=max_inner_pause_ms, hard_fade_ms=hard_fade_ms),
               dropped_pieces=dropped, units_without_audio=no_audio, units=log_units)
    with open(cuts_log_path(output), "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    progress.done(f"wrote {output} — {fmt_ts(len(result) / 1000)} from {len(kept)} spans in {len(plan)} pieces "
                  f"({inner_cuts} inner pauses shortened, {len(dropped)} silent pieces dropped, {hard_joins} hard "
                  f"joins, {in_sound} cuts that had to land in sound); cuts in {cuts_log_path(output)}")
    return positions
