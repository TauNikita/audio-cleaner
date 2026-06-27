"""Text normalization and script splitting.

The same `normalize()` is applied to both the script and the transcribed words so that fuzzy
matching compares apples to apples. Splitting turns the script into an ordered list of sentences,
each flagged with whether it ends a paragraph (paragraph ends get a longer pause later).
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
    paragraph_end: bool = False     # True if this is the last sentence of its paragraph


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


def split_script(text: str) -> list[Sentence]:
    """Turn raw script text into ordered, paragraph-aware sentences."""
    text = text.lstrip("﻿")  # drop a leading byte-order mark if the file has one
    sentences: list[Sentence] = []
    for paragraph in _split_paragraphs(text):
        para_sentences = _split_sentences(paragraph)
        for i, raw in enumerate(para_sentences):
            sentences.append(
                Sentence(
                    index=len(sentences) + 1,
                    raw=raw,
                    norm=normalize(raw),
                    tokens=tokenize(raw),
                    paragraph_end=(i == len(para_sentences) - 1),
                )
            )
    return sentences


def looks_like_prose(sentences: list[Sentence]) -> bool:
    """Heuristic: real prose has a healthy average sentence length.

    Bullet points and shorthand produce many tiny fragments, which align poorly.
    """
    if not sentences:
        return False
    avg_tokens = sum(len(s.tokens) for s in sentences) / len(sentences)
    return avg_tokens >= 5
