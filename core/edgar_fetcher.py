"""
EDGAR 10-K fetcher for the multi-vector embedding pipeline.

Fetches SEC 10-K filings via the `edgartools` library and returns chunks
in the same format as chunker.process_markdown.
"""

from __future__ import annotations

import logging

from edgar import Company, set_identity

from core.chunker import parse_sections, estimate_tokens

logger = logging.getLogger(__name__)

EDGAR_IDENTITY = "EmbeddingPipeline pipeline@example.com"

_identity_set = False

TENK_SECTIONS = [
    "Item 1",
    "Item 1A",
    "Item 1B",
    "Item 2",
    "Item 3",
    "Item 4",
    "Item 5",
    "Item 6",
    "Item 7",
    "Item 7A",
    "Item 8",
    "Item 9",
    "Item 9A",
    "Item 9B",
]


def _ensure_identity() -> None:
    """Call set_identity() once per process."""
    global _identity_set
    if not _identity_set:
        set_identity(EDGAR_IDENTITY)
        _identity_set = True


def fetch_tenk_sections(ticker: str) -> list[dict]:
    """Fetch sections from the latest 10-K filing for *ticker*.

    Returns ``(sections, metadata)`` where *sections* is a list of
    ``{"section": str, "text": str}`` dicts (same format as
    ``chunker.parse_sections``) and *metadata* contains company name,
    filing date, ticker, and form type.

    Raises ``ValueError`` if the company or filing cannot be found.
    """
    _ensure_identity()

    try:
        company = Company(ticker)
    except Exception as exc:
        raise ValueError(
            f"Could not find company for ticker '{ticker}': {exc}"
        ) from exc

    filings = company.get_filings(form="10-K")
    filing = filings.latest() if filings else None
    if filing is None:
        raise ValueError(
            f"No 10-K filings found for ticker '{ticker}'"
        )

    tenk = filing.obj()

    # Try section-level extraction first
    sections: list[dict] = []
    for item_name in TENK_SECTIONS:
        try:
            section_obj = tenk[item_name]
            text = section_obj.text()
            if text and text.strip():
                sections.append({"section": item_name, "text": text.strip()})
        except Exception:
            continue

    # If section-level parsing got very little, fall back to markdown
    if len(sections) < 3:
        logger.info(
            "Section-level parsing found only %d sections for %s, "
            "falling back to markdown extraction",
            len(sections), ticker,
        )
        try:
            markdown = filing.markdown()
            # parse_sections splits on # and ## headers — the markdown
            # from EDGAR uses ### and #### so we promote them
            markdown = markdown.replace("#### ", "## ").replace("### ", "# ")
            sections = parse_sections(markdown)
            # Filter out tiny sections (bullet fragments, etc.)
            sections = [s for s in sections if len(s["text"].split()) > 50]
            logger.info(
                "Markdown fallback produced %d sections for %s",
                len(sections), ticker,
            )
        except Exception as exc:
            logger.warning("Markdown fallback failed for %s: %s", ticker, exc)

    metadata = {
        "ticker": ticker.upper(),
        "company_name": str(getattr(company, "name", ticker)),
        "filing_date": str(getattr(filing, "filing_date", "")),
        "form_type": "10-K",
    }

    return sections, metadata


def process_ticker(ticker: str) -> tuple[list[dict], dict]:
    """Fetch the latest 10-K for *ticker* and return sections as-is.

    No pre-chunking — each section is sent whole to the embedding model.
    Sections exceeding the model's 8192 token limit will be truncated
    by the API, but the model sees full context for what it does embed.

    Returns ``(chunks_list, metadata_dict)`` where each chunk has keys
    ``section``, ``text``, ``chunk_index``, ``token_count``.
    """
    sections, metadata = fetch_tenk_sections(ticker)

    all_chunks: list[dict] = []
    for idx, sec in enumerate(sections):
        all_chunks.append({
            "section": sec["section"],
            "text": sec["text"],
            "chunk_index": 0,
            "token_count": estimate_tokens(sec["text"]),
        })

    return all_chunks, metadata
