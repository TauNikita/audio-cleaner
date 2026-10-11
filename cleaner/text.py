"""Text normalization and script splitting.

The same `normalize()` is applied to both the script and the transcribed words so that fuzzy
matching compares apples to apples. Splitting turns the script into an ordered list of units,
each flagged with whether it ends a sentence and/or a paragraph (those get pauses later). A unit is
a sentence by default. With `clauses=True` sentences are further split at commas, semicolons,
colons and dashes, which picks up a retake from the last comma, but every extra cut is a chance to
clip a word ending, so it is opt-in.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Apostrophe-like characters we fold to a plain ASCII apostrophe so "Ukraine's" survives intact.
_APOSTROPHES = "’ʼ‘"


def normalize(text: str) -> str:
    """Lowercase, drop punctuation but keep apostrophes, collapse whitespace.

    Hyphens become spaces ("green-and-yellow" -> "green and yellow") to line up with how speech is
    transcribed. Anything that isn't a letter, digit, apostrophe or space is treated as a separator.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    for ch in _APOSTROPHES:
        text = text.replace(ch, "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    # Drop stray apostrophes that aren't gluing letters together (e.g. quotes turned into ').
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalized whitespace tokens."""
    norm = normalize(text)
    return norm.split() if norm else []


@dataclass
class Sentence:
    index: int                      # 1-based position in the script
    raw: str                        # original text, trimmed
    norm: str                       # normalized text
    tokens: list[str] = field(default_factory=list)
    paragraph_end: bool = False     # True if this is the last unit of its paragraph
    sentence_end: bool = True       # False for a clause that is followed by more of its sentence


def _split_paragraphs(text: str) -> list[str]:
    """Split into paragraph blocks.

    Prefer blank-line separation. If the text has no blank lines (a common shape where each
    paragraph sits on its own single line), fall back to treating every non-empty line as a
    paragraph.
    """
    blocks = [b.strip() for b in re.split(r"\n[ \t]*\n", text) if b.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else blocks


# A sentence boundary is sentence-ending punctuation followed by whitespace and the start of the
# next sentence (a capital letter or an opening quote). Requiring whitespace after the punctuation
# keeps decimals like "96.77" intact, and requiring a capital/quote next avoids splitting inside
# abbreviations such as "a.m. on ...".
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])["\']?\s+(?=["\'(]?[A-Z])')


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


# A clause boundary: a comma, semicolon or colon followed by whitespace, or a spaced dash. The
# punctuation stays with the clause before it.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+|\s+[—–-]\s+")

MIN_CLAUSE_WORDS = 4


def _split_clauses(sentence: str, min_words: int = MIN_CLAUSE_WORDS) -> list[str]:
    """Split a sentence at clause punctuation, folding pieces shorter than min_words into their
    neighbour so no unit is too short to align reliably."""
    parts = [p.strip() for p in _CLAUSE_BOUNDARY.split(sentence) if p and p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and (len(tokenize(part)) < min_words or len(tokenize(merged[-1])) < min_words):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and len(tokenize(merged[-1])) < min_words:
        last = merged.pop()
        merged[-1] = f"{merged[-1]} {last}"
    return merged or [sentence]


def split_script(text: str, clauses: bool = False) -> list[Sentence]:
    """Turn raw script text into ordered units: whole sentences by default, clauses on request."""
    text = text.lstrip("﻿")  # drop a leading byte-order mark if the file has one
    units: list[Sentence] = []
    for paragraph in _split_paragraphs(text):
        para_sentences = _split_sentences(paragraph)
        for i, sentence in enumerate(para_sentences):
            pieces = _split_clauses(sentence) if clauses else [sentence]
            for j, raw in enumerate(pieces):
                last = j == len(pieces) - 1
                units.append(
                    Sentence(
                        index=len(units) + 1,
                        raw=raw,
                        norm=normalize(raw),
                        tokens=tokenize(raw),
                        paragraph_end=last and i == len(para_sentences) - 1,
                        sentence_end=last,
                    )
                )
    return units


def looks_like_prose(sentences: list[Sentence]) -> bool:
    """Heuristic: real prose has a healthy average sentence length.

    Bullet points and shorthand produce many tiny fragments, which align poorly.
    """
    if not sentences:
        return False
    avg_tokens = sum(len(s.tokens) for s in sentences) / len(sentences)
    return avg_tokens >= 5
