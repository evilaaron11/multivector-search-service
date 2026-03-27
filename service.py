"""Orchestration layer for the multi-vector embedding pipeline.

Phase 6: ties together chunking, storage, embedding, and search into
clean public functions that return dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core import store, chunker, embedder, searcher, edgar_fetcher


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    document_id: int
    filename: str
    num_chunks: int
    sections: list[str]


@dataclass
class SearchResult:
    chunk_id: int
    score: float
    text: str
    section: str
    document_id: int
    chunk_index: int
    highlights: list[tuple[str, float, str]]


@dataclass
class DocumentInfo:
    document_id: int
    filename: str
    ingested_at: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(file_path: str) -> IngestResult:
    """Parse, chunk, embed, and store a markdown document.

    Returns an IngestResult summarising what was ingested.
    """
    store.init_db()

    # Parse and chunk
    chunks = chunker.process_markdown(file_path)

    # Insert document row
    filename = Path(file_path).name
    document_id = store.insert_document(filename)

    # Insert each chunk and collect ids / texts
    chunk_ids: list[int] = []
    chunk_texts: list[str] = []
    for chunk in chunks:
        cid = store.insert_chunk(
            document_id=document_id,
            text=chunk["text"],
            section=chunk["section"],
            chunk_index=chunk["chunk_index"],
            token_count=chunk["token_count"],
        )
        chunk_ids.append(cid)
        chunk_texts.append(chunk["text"])

    # Embed all chunk texts in one call (internally batched)
    embeddings = embedder.embed_documents(chunk_texts)

    # Save each embedding vector
    for cid, vec in zip(chunk_ids, embeddings):
        store.save_vectors(cid, vec)

    # Collect unique sections (preserve insertion order)
    seen: set[str] = set()
    sections: list[str] = []
    for chunk in chunks:
        sec = chunk["section"]
        if sec not in seen:
            seen.add(sec)
            sections.append(sec)

    return IngestResult(
        document_id=document_id,
        filename=filename,
        num_chunks=len(chunks),
        sections=sections,
    )


def ingest_ticker(ticker: str) -> IngestResult:
    """Fetch SEC filing for a ticker, chunk, embed, and store it.

    Returns an IngestResult summarising what was ingested.
    """
    store.init_db()

    # Fetch and chunk the filing
    chunks, metadata = edgar_fetcher.process_ticker(ticker)

    # Build a descriptive filename from metadata
    filename = f"{metadata['ticker']}_10K_{metadata['filing_date']}"

    # Insert document row
    document_id = store.insert_document(filename)

    # Insert each chunk and collect ids / texts
    chunk_ids: list[int] = []
    chunk_texts: list[str] = []
    for chunk in chunks:
        cid = store.insert_chunk(
            document_id=document_id,
            text=chunk["text"],
            section=chunk["section"],
            chunk_index=chunk["chunk_index"],
            token_count=chunk["token_count"],
        )
        chunk_ids.append(cid)
        chunk_texts.append(chunk["text"])

    # Embed all chunk texts in one call (internally batched)
    embeddings = embedder.embed_documents(chunk_texts)

    # Save each embedding vector
    for cid, vec in zip(chunk_ids, embeddings):
        store.save_vectors(cid, vec)

    # Collect unique sections (preserve insertion order)
    seen: set[str] = set()
    sections: list[str] = []
    for chunk in chunks:
        sec = chunk["section"]
        if sec not in seen:
            seen.add(sec)
            sections.append(sec)

    return IngestResult(
        document_id=document_id,
        filename=filename,
        num_chunks=len(chunks),
        sections=sections,
    )


def search_documents(
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
    section: str | None = None,
) -> list[SearchResult]:
    """Search ingested documents and return ranked results."""
    store.init_db()

    raw_results = searcher.search(
        query=query,
        top_k=top_k,
        document_id=document_id,
        section=section,
    )

    return [
        SearchResult(
            chunk_id=r["chunk_id"],
            score=r["score"],
            text=r["text"],
            section=r["section"],
            document_id=r["document_id"],
            chunk_index=r["chunk_index"],
            highlights=r.get("highlights", []),
        )
        for r in raw_results
    ]


def list_documents() -> list[DocumentInfo]:
    """Return metadata for every ingested document."""
    store.init_db()

    rows = store.list_documents()
    return [
        DocumentInfo(
            document_id=r["id"],
            filename=r["filename"],
            ingested_at=r["ingested_at"],
            chunk_count=r["chunk_count"],
        )
        for r in rows
    ]


def delete_document(document_id: int) -> bool:
    """Delete a document and all associated data.

    Returns True if the document existed, False otherwise.
    """
    store.init_db()

    doc = store.get_document(document_id)
    if doc is None:
        return False

    store.delete_document(document_id)
    return True
