"""
Markdown parser and chunker.

Splits markdown documents into sections, then chunks each section into
overlapping windows of a target token size for optimal embedding quality.
"""

from __future__ import annotations

import re
from pathlib import Path

API_TOKEN_LIMIT = 8192
CHUNK_TARGET = 256
CHUNK_MAX = 512
CHUNK_OVERLAP = 64


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count via whitespace split. Good enough for POC."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Markdown section parsing
# ---------------------------------------------------------------------------

def parse_sections(markdown_text: str) -> list[dict]:
    """Parse markdown into sections delimited by ``#`` and ``##`` headers.

    Returns a list of ``{"section": str, "text": str}`` dicts.

    * Text before the first header is placed in a **"Preamble"** section.
    * Consecutive headers with no content between them collapse so that only
      the *last* header is kept (the earlier ones are discarded).
    * Trailing whitespace is stripped from every section's text.
    """
    header_pattern = re.compile(r"^(#{1,2})\s+(.*)", re.MULTILINE)

    # Find every header and its position
    matches = list(header_pattern.finditer(markdown_text))

    if not matches:
        # No headers at all – everything is preamble
        text = markdown_text.strip()
        if text:
            return [{"section": "Preamble", "text": text}]
        return []

    sections: list[dict] = []

    # Handle preamble (text before the first header)
    preamble_text = markdown_text[: matches[0].start()].strip()
    if preamble_text:
        sections.append({"section": "Preamble", "text": preamble_text})

    # Walk through headers and collect the text that follows each one.
    for i, match in enumerate(matches):
        header_title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[body_start:body_end].strip()

        if not body and i + 1 < len(matches):
            # Consecutive header with no content – skip it; the next header
            # will become the effective section header.
            continue

        sections.append({"section": header_title, "text": body})

    return sections


# ---------------------------------------------------------------------------
# Chunking with overlap
# ---------------------------------------------------------------------------

def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double-newline separated).

    Falls back to single-newline splits, then sentence-level if paragraphs
    are too large.
    """
    # Try double-newline first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if paragraphs:
        return paragraphs

    # Fallback: single newlines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines if lines else [text]


def chunk_section(
    section: str,
    text: str,
    target: int = CHUNK_TARGET,
    maximum: int = CHUNK_MAX,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split a section into overlapping chunks.

    Strategy:
    1. Split text into paragraphs.
    2. Accumulate paragraphs until reaching the target size.
    3. If a single paragraph exceeds the max, hard-split it on word boundaries.
    4. Each new chunk starts with `overlap` tokens from the end of the previous chunk.

    Returns a list of dicts::
        {"section": str, "text": str, "chunk_index": int, "token_count": int}
    """
    text = text.strip()
    if not text:
        return []

    total = estimate_tokens(text)

    # Small enough for a single chunk — no splitting needed
    if total <= maximum:
        return [
            {
                "section": section,
                "text": text,
                "chunk_index": 0,
                "token_count": total,
            }
        ]

    paragraphs = _split_into_paragraphs(text)

    # Build chunks by accumulating paragraphs
    chunks: list[dict] = []
    current_words: list[str] = []
    overlap_words: list[str] = []

    def _flush():
        if not current_words:
            return
        chunk_text = " ".join(current_words)
        chunks.append({
            "section": section,
            "text": chunk_text,
            "chunk_index": len(chunks),
            "token_count": len(current_words),
        })

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        # If this single paragraph exceeds the max, hard-split it
        if len(para_words) > maximum:
            # Flush what we have first
            _flush()
            # Hard-split the oversized paragraph
            for i in range(0, len(para_words), target):
                if chunks and i == 0:
                    # Start with overlap from previous chunk
                    chunk_words = overlap_words + para_words[i : i + target]
                else:
                    start = max(0, i - overlap) if i > 0 else 0
                    chunk_words = para_words[start : i + target]
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "section": section,
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "token_count": len(chunk_words),
                })
            overlap_words = para_words[-overlap:] if len(para_words) >= overlap else para_words[:]
            current_words = []
            continue

        # Would adding this paragraph exceed the target?
        if len(current_words) + len(para_words) > target and current_words:
            _flush()
            # Start new chunk with overlap from previous
            overlap_words = current_words[-overlap:] if len(current_words) >= overlap else current_words[:]
            current_words = overlap_words + para_words
        else:
            # First chunk or still under target — prepend overlap if starting fresh
            if not current_words and overlap_words:
                current_words = overlap_words + para_words
            else:
                current_words.extend(para_words)

    # Flush remaining
    _flush()

    return chunks


# ---------------------------------------------------------------------------
# End-to-end file processing
# ---------------------------------------------------------------------------

def process_markdown(file_path: str) -> list[dict]:
    """Read a ``.md`` file, parse sections, chunk each section, and return a
    flat list of all chunk dicts."""
    content = Path(file_path).read_text(encoding="utf-8")
    sections = parse_sections(content)

    all_chunks: list[dict] = []
    for sec in sections:
        all_chunks.extend(chunk_section(sec["section"], sec["text"]))

    return all_chunks
