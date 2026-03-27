"""Unit tests for core.store — SQLite + NumPy storage layer."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_config(tmp_path, monkeypatch):
    """Redirect DB_PATH and VECTORS_DIR to a temporary directory."""
    db_path = tmp_path / "test.db"
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()

    import config
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "VECTORS_DIR", vectors_dir)

    # Also patch the module-level references that store.py already imported.
    # Re-import store so it picks up the patched values through config.
    import importlib
    import core.store as store_mod
    importlib.reload(store_mod)

    monkeypatch.setattr(store_mod, "DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "VECTORS_DIR", vectors_dir)


@pytest.fixture
def store():
    from core.store import (
        init_db,
        insert_document,
        insert_chunk,
        get_document,
        get_chunk,
        get_chunks_by_document,
        get_all_chunk_ids,
        list_documents,
        delete_document,
        save_vectors,
        load_vectors,
        delete_vectors,
    )
    import core.store as mod
    init_db()
    return mod


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(store):
    """init_db should create the documents and chunks tables."""
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "documents" in tables
    assert "chunks" in tables


def test_init_db_is_idempotent(store):
    """Calling init_db a second time should not raise."""
    store.init_db()


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

def test_insert_and_get_document(store):
    doc_id = store.insert_document("report.pdf")
    assert isinstance(doc_id, int)

    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc["filename"] == "report.pdf"
    assert "ingested_at" in doc


def test_get_document_nonexistent(store):
    assert store.get_document(9999) is None


def test_list_documents_empty(store):
    assert store.list_documents() == []


def test_list_documents_with_chunk_counts(store):
    d1 = store.insert_document("a.pdf")
    d2 = store.insert_document("b.pdf")
    store.insert_chunk(d1, "hello", "intro", 0, 10)
    store.insert_chunk(d1, "world", "intro", 1, 10)
    store.insert_chunk(d2, "foo", "body", 0, 5)

    docs = store.list_documents()
    assert len(docs) == 2

    by_id = {d["id"]: d for d in docs}
    assert by_id[d1]["chunk_count"] == 2
    assert by_id[d2]["chunk_count"] == 1


def test_list_documents_zero_chunks(store):
    store.insert_document("empty.pdf")
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 0


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------

def test_insert_and_get_chunk(store):
    doc_id = store.insert_document("doc.txt")
    chunk_id = store.insert_chunk(doc_id, "sample text", "intro", 0, 42)
    assert isinstance(chunk_id, int)

    chunk = store.get_chunk(chunk_id)
    assert chunk is not None
    assert chunk["document_id"] == doc_id
    assert chunk["text"] == "sample text"
    assert chunk["section"] == "intro"
    assert chunk["chunk_index"] == 0
    assert chunk["token_count"] == 42


def test_get_chunk_nonexistent(store):
    assert store.get_chunk(9999) is None


def test_get_chunks_by_document(store):
    doc_id = store.insert_document("doc.txt")
    store.insert_chunk(doc_id, "first", "s1", 0, 5)
    store.insert_chunk(doc_id, "second", "s1", 1, 6)

    chunks = store.get_chunks_by_document(doc_id)
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_get_chunks_by_document_empty(store):
    doc_id = store.insert_document("empty.txt")
    assert store.get_chunks_by_document(doc_id) == []


# ---------------------------------------------------------------------------
# get_all_chunk_ids filtering
# ---------------------------------------------------------------------------

def test_get_all_chunk_ids_no_filter(store):
    d1 = store.insert_document("a.txt")
    d2 = store.insert_document("b.txt")
    c1 = store.insert_chunk(d1, "t", "intro", 0, 1)
    c2 = store.insert_chunk(d2, "t", "body", 0, 1)

    ids = store.get_all_chunk_ids()
    assert c1 in ids
    assert c2 in ids


def test_get_all_chunk_ids_filter_by_document(store):
    d1 = store.insert_document("a.txt")
    d2 = store.insert_document("b.txt")
    c1 = store.insert_chunk(d1, "t", "intro", 0, 1)
    store.insert_chunk(d2, "t", "body", 0, 1)

    ids = store.get_all_chunk_ids(document_id=d1)
    assert ids == [c1]


def test_get_all_chunk_ids_filter_by_section(store):
    d1 = store.insert_document("a.txt")
    c1 = store.insert_chunk(d1, "t", "intro", 0, 1)
    c2 = store.insert_chunk(d1, "t", "body", 1, 1)

    ids = store.get_all_chunk_ids(section="body")
    assert ids == [c2]


def test_get_all_chunk_ids_filter_both(store):
    d1 = store.insert_document("a.txt")
    d2 = store.insert_document("b.txt")
    store.insert_chunk(d1, "t", "intro", 0, 1)
    c2 = store.insert_chunk(d1, "t", "body", 1, 1)
    store.insert_chunk(d2, "t", "body", 0, 1)

    ids = store.get_all_chunk_ids(document_id=d1, section="body")
    assert ids == [c2]


# ---------------------------------------------------------------------------
# delete_document cascade
# ---------------------------------------------------------------------------

def test_delete_document_cascades(store):
    doc_id = store.insert_document("bye.txt")
    c1 = store.insert_chunk(doc_id, "a", "s", 0, 1)
    c2 = store.insert_chunk(doc_id, "b", "s", 1, 1)

    # Save vectors for both chunks
    arr = np.random.rand(4, 128).astype(np.float32)
    store.save_vectors(c1, arr)
    store.save_vectors(c2, arr)

    store.delete_document(doc_id)

    assert store.get_document(doc_id) is None
    assert store.get_chunks_by_document(doc_id) == []
    assert store.get_chunk(c1) is None
    assert store.get_chunk(c2) is None

    # Vector files should also be gone
    from config import VECTORS_DIR
    assert not (VECTORS_DIR / f"{c1}.npz").exists()
    assert not (VECTORS_DIR / f"{c2}.npz").exists()


# ---------------------------------------------------------------------------
# Vector file I/O
# ---------------------------------------------------------------------------

def test_save_and_load_vectors_roundtrip(store):
    doc_id = store.insert_document("v.txt")
    chunk_id = store.insert_chunk(doc_id, "vec", "s", 0, 1)

    original = np.random.rand(10, 128).astype(np.float32)
    store.save_vectors(chunk_id, original)
    loaded = store.load_vectors(chunk_id)

    np.testing.assert_array_equal(original, loaded)


def test_delete_vectors_removes_file(store):
    doc_id = store.insert_document("v.txt")
    chunk_id = store.insert_chunk(doc_id, "vec", "s", 0, 1)

    store.save_vectors(chunk_id, np.array([[1.0, 2.0]]))
    store.delete_vectors(chunk_id)

    from config import VECTORS_DIR
    assert not (VECTORS_DIR / f"{chunk_id}.npz").exists()


def test_delete_vectors_noop_when_missing(store):
    """delete_vectors should not raise if the file does not exist."""
    store.delete_vectors(99999)
