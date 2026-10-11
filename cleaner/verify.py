"""Check the rendered output against the script, sentence by sentence.

Each kept sentence is re-transcribed from the output *in isolation* and compared with its script
text, and its duration is compared with what the speaker's own pace predicts. That catches what the
alignment report cannot: a retake that slipped through, a clipped first word, a sentence that is
far longer than its words. Findings are printed with output timestamps so they can be listened to.
"""

from .align import Decision
from .hidden import Energy, Prober, LONG_WORD_S
from .util import Progress, fmt_ts

MIN_SCORE = 80          # fuzzy ratio below which the heard text no longer matches the sentence
EXTRA_WORDS = 3         # this many words more than the script says is a leak, not ASR noise
SLOW_FACTOR = 1.35      # a sentence this much longer than pace predicts (plus 1 s) hides something


def verify(output_wav16k: str, positions: list[tuple[Decision, float, float]],
           prober: Prober) -> int:
    """Print a report of suspicious sentences in the output. Returns how many were flagged."""
    from rapidfuzz import fuzz

    energy = Energy(output_wav16k)
    progress = Progress("Verify")
    rows = []
    for k, (d, a, b) in enumerate(positions):
        heard = prober.words(energy, a - 0.05, b + 0.05)
        text = " ".join(h.norm for h in heard)
        score = fuzz.ratio(d.sentence.norm, text)
        rows.append((d, a, b, heard, text, score))
        progress.update(f"{k + 1}/{len(positions)} sentences")
    progress.done(f"done — {len(positions)} sentences re-transcribed from the output")

    # Pace from the clean sentences, so a slow reader isn't flagged everywhere.
    rates = sorted(len(d.sentence.tokens) / max(0.5, b - a)
                   for d, a, b, _, _, score in rows if score >= 90)
    rate = rates[len(rates) // 2] if rates else 2.5

    flagged = 0
    for d, a, b, heard, text, score in rows:
        reasons = []
        extra = len(heard) - len(d.sentence.tokens)
        expected = len(d.sentence.tokens) / rate
        if score < MIN_SCORE:
            reasons.append(f"match {score:.0f}")
        if extra >= EXTRA_WORDS:
            reasons.append(f"{extra} extra words")
        if b - a > expected * SLOW_FACTOR + 1.0:
            reasons.append(f"{b - a:.1f}s for {expected:.1f}s of words")
        # The first word's start is pinned to the window edge, so only a gross stretch counts there.
        # Pauses stretch whisper's word timings a little; only a gross stretch means hidden speech.
        long_words = [h for i, h in enumerate(heard) if h.end - h.start > LONG_WORD_S + 0.6]
        if long_words:
            reasons.append("stretched word at " + ", ".join(fmt_ts(h.start) for h in long_words[:3]))
        if not reasons:
            continue
        flagged += 1
        if flagged == 1:
            print("\nVerify: sentences worth a listen (output time):")
        print(f"  {fmt_ts(a):>9}-{fmt_ts(b):<9} #{d.sentence.index:<3} {'; '.join(reasons)}")
        print(f"            script: {' '.join(d.sentence.tokens)[:110]}")
        print(f"            heard : {text[:110]}")
    if flagged:
        print(f"Verify: {flagged}/{len(rows)} sentences flagged.\n")
    else:
        print("Verify: every sentence in the output matches the script at the speaker's pace.\n")
    return flagged
