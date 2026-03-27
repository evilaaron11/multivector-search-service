"""Unit tests for the service orchestration layer.

All underlying modules (store, chunker, embedder, searcher) are mocked
so no real DB, API, or filesystem access occurs.
"""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock, call
import numpy as np
import pytest

import service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_CHUNKS = [
    {
        "section": "Item 1",
        "text": "First chunk text.",
        "chunk_index": 0,
        "token_count": 3,
    },
    {
        "section": "Item 1",
        "text": "Second chunk text.",
        "chunk_index": 1,
        "token_count": 3,
    },
    {
        "section": "Item 7",
        "text": "Third chunk text.",
        "chunk_index": 0,
        "token_count": 3,
    },
]

FAKE_EMBEDDINGS = [
    np.zeros((5, 128), dtype=np.float32),
    np.ones((5, 128), dtype=np.float32),
    np.full((5, 128), 2.0, dtype=np.float32),
]

FAKE_SEARCH_RESULTS = [
    {
        "chunk_id": 10,
        "score": 0.95,
        "text": "Matching chunk text.",
        "section": "Item 7",
        "document_id": 1,
        "chunk_index": 0,
        "token_count": 3,
    },
    {
        "chunk_id": 11,
        "score": 0.80,
        "text": "Another matching chunk.",
        "section": "Item 1",
        "document_id": 1,
        "chunk_index": 1,
        "token_count": 3,
    },
]

FAKE_DOC_ROWS = [
    {
        "id": 1,
        "filename": "report.md",
        "ingested_at": "2026-01-15T10:30:00",
        "chunk_count": 12,
    },
    {
        "id": 2,
        "filename": "filing.md",
        "ingested_at": "2026-02-20T14:00:00",
        "chunk_count": 8,
    },
]


# ---------------------------------------------------------------------------
# ingest_document tests
# ---------------------------------------------------------------------------

@patch("service.embedder")
@patch("service.store")
@patch("service.chunker")
def test_ingest_calls_in_correct_order(mock_chunker, mock_store, mock_embedder):
    """Verify that ingest calls chunker, store inserts, embedder, and
    save_vectors in the expected sequence."""
    mock_chunker.process_markdown.return_value = FAKE_CHUNKS
    mock_store.insert_document.return_value = 42
    mock_store.insert_chunk.side_effect = [100, 101, 102]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    service.ingest_document("/tmp/test.md")

    # 1. init_db
    mock_store.init_db.assert_called_once()

    # 2. chunker processes the file
    mock_chunker.process_markdown.assert_called_once_with("/tmp/test.md")

    # 3. document inserted
    mock_store.insert_document.assert_called_once_with("test.md")

    # 4. each chunk inserted
    assert mock_store.insert_chunk.call_count == 3
    mock_store.insert_chunk.assert_any_call(
        document_id=42,
        text="First chunk text.",
        section="Item 1",
        chunk_index=0,
        token_count=3,
    )

    # 5. embedder called with all chunk texts
    mock_embedder.embed_documents.assert_called_once_with(
        ["First chunk text.", "Second chunk text.", "Third chunk text."]
    )

    # 6. save_vectors called for each chunk
    assert mock_store.save_vectors.call_count == 3
    mock_store.save_vectors.assert_any_call(100, FAKE_EMBEDDINGS[0])
    mock_store.save_vectors.assert_any_call(101, FAKE_EMBEDDINGS[1])
    mock_store.save_vectors.assert_any_call(102, FAKE_EMBEDDINGS[2])


@patch("service.embedder")
@patch("service.store")
@patch("service.chunker")
def test_ingest_returns_correct_result(mock_chunker, mock_store, mock_embedder):
    """Verify IngestResult fields are populated correctly."""
    mock_chunker.process_markdown.return_value = FAKE_CHUNKS
    mock_store.insert_document.return_value = 7
    mock_store.insert_chunk.side_effect = [10, 11, 12]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    result = service.ingest_document("/data/report.md")

    assert isinstance(result, service.IngestResult)
    assert result.document_id == 7
    assert result.filename == "report.md"
    assert result.num_chunks == 3
    assert result.sections == ["Item 1", "Item 7"]


# ---------------------------------------------------------------------------
# search_documents tests
# ---------------------------------------------------------------------------

@patch("service.searcher")
@patch("service.store")
def test_search_calls_searcher(mock_store, mock_searcher):
    """Verify search_documents delegates to searcher.search and returns
    SearchResult objects."""
    mock_searcher.search.return_value = FAKE_SEARCH_RESULTS

    results = service.search_documents("revenue growth", top_k=2)

    mock_store.init_db.assert_called_once()
    mock_searcher.search.assert_called_once_with(
        query="revenue growth",
        top_k=2,
        document_id=None,
        section=None,
    )

    assert len(results) == 2
    assert all(isinstance(r, service.SearchResult) for r in results)
    assert results[0].chunk_id == 10
    assert results[0].score == 0.95
    assert results[1].text == "Another matching chunk."


@patch("service.searcher")
@patch("service.store")
def test_search_passes_filters(mock_store, mock_searcher):
    """Verify document_id and section filters are forwarded to searcher."""
    mock_searcher.search.return_value = []

    service.search_documents(
        "risk factors",
        top_k=3,
        document_id=5,
        section="Item 1A",
    )

    mock_searcher.search.assert_called_once_with(
        query="risk factors",
        top_k=3,
        document_id=5,
        section="Item 1A",
    )


# ---------------------------------------------------------------------------
# list_documents tests
# ---------------------------------------------------------------------------

@patch("service.store")
def test_list_documents_returns_document_info(mock_store):
    """Verify list_documents returns DocumentInfo objects."""
    mock_store.list_documents.return_value = FAKE_DOC_ROWS

    results = service.list_documents()

    mock_store.init_db.assert_called_once()
    assert len(results) == 2
    assert all(isinstance(d, service.DocumentInfo) for d in results)
    assert results[0].document_id == 1
    assert results[0].filename == "report.md"
    assert results[0].chunk_count == 12
    assert results[1].ingested_at == "2026-02-20T14:00:00"


# ---------------------------------------------------------------------------
# delete_document tests
# ---------------------------------------------------------------------------

@patch("service.store")
def test_delete_document_exists(mock_store):
    """Verify delete returns True when the document exists."""
    mock_store.get_document.return_value = {"id": 1, "filename": "f.md"}

    result = service.delete_document(1)

    mock_store.init_db.assert_called_once()
    mock_store.get_document.assert_called_once_with(1)
    mock_store.delete_document.assert_called_once_with(1)
    assert result is True


@patch("service.store")
def test_delete_document_not_found(mock_store):
    """Verify delete returns False when the document does not exist."""
    mock_store.get_document.return_value = None

    result = service.delete_document(999)

    mock_store.init_db.assert_called_once()
    mock_store.get_document.assert_called_once_with(999)
    mock_store.delete_document.assert_not_called()
    assert result is False


# ---------------------------------------------------------------------------
# ingest_ticker tests
# ---------------------------------------------------------------------------

FAKE_TICKER_METADATA = {
    "ticker": "INTC",
    "company_name": "Intel Corporation",
    "filing_date": "2024-02-21",
    "form_type": "10-K",
}


@patch("service.embedder")
@patch("service.store")
@patch("service.edgar_fetcher")
def test_ingest_ticker_calls_process_ticker(mock_edgar, mock_store, mock_embedder):
    """Verify ingest_ticker calls edgar_fetcher.process_ticker with the ticker."""
    mock_edgar.process_ticker.return_value = (FAKE_CHUNKS, FAKE_TICKER_METADATA)
    mock_store.insert_document.return_value = 1
    mock_store.insert_chunk.side_effect = [100, 101, 102]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    service.ingest_ticker("INTC")

    mock_edgar.process_ticker.assert_called_once_with("INTC")


@patch("service.embedder")
@patch("service.store")
@patch("service.edgar_fetcher")
def test_ingest_ticker_creates_correct_filename(mock_edgar, mock_store, mock_embedder):
    """Verify the generated filename uses ticker, form type, and filing date."""
    mock_edgar.process_ticker.return_value = (FAKE_CHUNKS, FAKE_TICKER_METADATA)
    mock_store.insert_document.return_value = 1
    mock_store.insert_chunk.side_effect = [100, 101, 102]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    result = service.ingest_ticker("INTC")

    assert result.filename == "INTC_10K_2024-02-21"
    mock_store.insert_document.assert_called_once_with("INTC_10K_2024-02-21")


@patch("service.embedder")
@patch("service.store")
@patch("service.edgar_fetcher")
def test_ingest_ticker_returns_correct_result(mock_edgar, mock_store, mock_embedder):
    """Verify IngestResult fields are populated correctly for ticker ingest."""
    mock_edgar.process_ticker.return_value = (FAKE_CHUNKS, FAKE_TICKER_METADATA)
    mock_store.insert_document.return_value = 55
    mock_store.insert_chunk.side_effect = [200, 201, 202]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    result = service.ingest_ticker("INTC")

    assert isinstance(result, service.IngestResult)
    assert result.document_id == 55
    assert result.filename == "INTC_10K_2024-02-21"
    assert result.num_chunks == 3
    assert result.sections == ["Item 1", "Item 7"]


@patch("service.embedder")
@patch("service.store")
@patch("service.edgar_fetcher")
def test_ingest_ticker_calls_in_correct_order(mock_edgar, mock_store, mock_embedder):
    """Verify that ingest_ticker calls store/embedder in the correct sequence."""
    mock_edgar.process_ticker.return_value = (FAKE_CHUNKS, FAKE_TICKER_METADATA)
    mock_store.insert_document.return_value = 42
    mock_store.insert_chunk.side_effect = [100, 101, 102]
    mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS

    service.ingest_ticker("INTC")

    # 1. init_db
    mock_store.init_db.assert_called_once()

    # 2. edgar_fetcher processes the ticker
    mock_edgar.process_ticker.assert_called_once_with("INTC")

    # 3. document inserted with generated filename
    mock_store.insert_document.assert_called_once_with("INTC_10K_2024-02-21")

    # 4. each chunk inserted
    assert mock_store.insert_chunk.call_count == 3
    mock_store.insert_chunk.assert_any_call(
        document_id=42,
        text="First chunk text.",
        section="Item 1",
        chunk_index=0,
        token_count=3,
    )

    # 5. embedder called with all chunk texts
    mock_embedder.embed_documents.assert_called_once_with(
        ["First chunk text.", "Second chunk text.", "Third chunk text."]
    )

    # 6. save_vectors called for each chunk
    assert mock_store.save_vectors.call_count == 3
    mock_store.save_vectors.assert_any_call(100, FAKE_EMBEDDINGS[0])
    mock_store.save_vectors.assert_any_call(101, FAKE_EMBEDDINGS[1])
    mock_store.save_vectors.assert_any_call(102, FAKE_EMBEDDINGS[2])


