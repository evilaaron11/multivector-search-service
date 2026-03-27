"""
Markdown parser and chunker for SEC 10-K/10-Q filings.

Splits markdown documents into sections, then hard-splits any section
that exceeds the Jina API token limit (8192 tokens).
"""

from __future__ import annotations

import re
from pathlib import Path

API_TOKEN_LIMIT = 8192


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
# Chunking
# ---------------------------------------------------------------------------

def chunk_section(section: str, text: str) -> list[dict]:
    """Split a section's text into chunks at the API token limit.

    If the section fits within the limit, returns it as a single chunk.
    Otherwise hard-splits on word boundaries.

    Returns a list of dicts::

        {"section": str, "text": str, "chunk_index": int, "token_count": int}
    """
    text = text.strip()
    if not text:
        return []

    total = estimate_tokens(text)

    if total <= API_TOKEN_LIMIT:
        return [
            {
                "section": section,
                "text": text,
                "chunk_index": 0,
                "token_count": total,
            }
        ]

    # Hard-split at the token limit
    words = text.split()
    chunks: list[dict] = []
    for i in range(0, len(words), API_TOKEN_LIMIT):
        chunk_words = words[i : i + API_TOKEN_LIMIT]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "section": section,
                "text": chunk_text,
                "chunk_index": len(chunks),
                "token_count": len(chunk_words),
            }
        )

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
