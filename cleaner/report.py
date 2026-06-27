"""Dry-run report: the gate where you sanity-check picks before any audio is rendered.

The score column makes a bad pick easy to spot (retakes aren't assumed to be in order), and MISSING /
low-confidence rows are flagged with concrete advice on how to fix them.
"""

from .align import Decision, LOW, MATCH, MISSING
from .util import fmt_ts

SNIPPET_WIDTH = 60


def _snippet(text: str, width: int = SNIPPET_WIDTH) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_report(decisions: list[Decision], threshold: float, model: str) -> dict:
    """Print the alignment table and a summary. Returns counts for the caller."""
    header = f"{'#':>4}  {'score':>5}  {'start':>11}  {'end':>11}  P  sentence"
    print()
    print(header)
    print("-" * len(header))

    counts = {MATCH: 0, LOW: 0, MISSING: 0}
    for d in decisions:
        counts[d.status] += 1
        idx = d.sentence.index
        para = "¶" if d.sentence.paragraph_end else " "
        if d.status == MISSING:
            print(f"{idx:>4}  {'--':>5}  {'MISSING':>11}  {'':>11}  {para}  "
                  f"!! {_snippet(d.sentence.raw)}")
        else:
            flag = "?" if d.status == LOW else " "
            print(f"{idx:>4}  {d.score:>5.0f}  {fmt_ts(d.start):>11}  {fmt_ts(d.end):>11}  "
                  f"{para}  {flag} {_snippet(d.sentence.raw)}")

    total = len(decisions)
    matched = counts[MATCH] + counts[LOW]
    print("-" * len(header))
    print(f"Summary: {matched}/{total} matched "
          f"({counts[MATCH]} solid, {counts[LOW]} low-confidence '?'), "
          f"{counts[MISSING]} MISSING '!!'.")

    if counts[LOW]:
        print(f"  '?' = only cleared a relaxed threshold (below {threshold:.0f}). "
              f"Listen to those spans before trusting them.")
    if counts[MISSING]:
        print("  '!!' = no take found. To fix MISSING sentences, try one or more of:")
        print(f"      - lower --threshold (currently {threshold:.0f})")
        print(f"      - use a bigger --model (currently '{model}', e.g. medium.en)")
        print("      - check the script sentence actually matches what you said")
    if not counts[MISSING] and not counts[LOW]:
        print("  All sentences matched cleanly.")
    print()
    return counts
