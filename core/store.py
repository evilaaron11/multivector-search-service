"""SQLite + NumPy storage layer for the multi-vector embedding pipeline."""

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DB_PATH, VECTORS_DIR


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the documents and chunks tables if they do not exist."""
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id  INTEGER NOT NULL REFERENCES documents(id),
                text         TEXT NOT NULL,
                section      TEXT NOT NULL,
                chunk_index  INTEGER NOT NULL,
                token_count  INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

def insert_document(filename: str) -> int:
    """Insert a document row and return its id."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO documents (filename, ingested_at) VALUES (?, ?)",
            (filename, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_document(document_id: int) -> dict | None:
    """Return a document as a dict, or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_documents() -> list[dict]:
    """Return all documents, each augmented with a chunk_count field."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT d.*, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(document_id: int) -> None:
    """Delete a document, its chunks, and all associated vector files."""
    chunk_ids = get_all_chunk_ids(document_id=document_id)
    for cid in chunk_ids:
        delete_vectors(cid)

    conn = _connect()
    try:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------

def insert_chunk(
    document_id: int,
    text: str,
    section: str,
    chunk_index: int,
    token_count: int,
) -> int:
    """Insert a chunk row and return its id."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO chunks (document_id, text, section, chunk_index, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, text, section, chunk_index, token_count),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_chunk(chunk_id: int) -> dict | None:
    """Return a chunk as a dict, or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_chunks_by_document(document_id: int) -> list[dict]:
    """Return all chunks belonging to a document, ordered by chunk_index."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_chunk_ids(
    document_id: int | None = None,
    section: str | None = None,
) -> list[int]:
    """Return chunk ids, optionally filtered by document_id and/or section."""
    query = "SELECT id FROM chunks"
    params: list = []
    clauses: list[str] = []

    if document_id is not None:
        clauses.append("document_id = ?")
        params.append(document_id)
    if section is not None:
        clauses.append("section = ?")
        params.append(section)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY id"

    conn = _connect()
    try:
        rows = conn.execute(query, params).fetchall()
        return [r["id"] for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Vector file I/O
# ---------------------------------------------------------------------------

def _vector_path(chunk_id: int) -> Path:
    return Path(VECTORS_DIR) / f"{chunk_id}.npz"


def save_vectors(chunk_id: int, vectors: np.ndarray) -> None:
    """Save a 2-D NumPy array to data/vectors/{chunk_id}.npz."""
    Path(VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(_vector_path(chunk_id)), vectors=vectors)


def load_vectors(chunk_id: int) -> np.ndarray:
    """Load a 2-D NumPy array from data/vectors/{chunk_id}.npz."""
    data = np.load(str(_vector_path(chunk_id)))
    return data["vectors"]


def delete_vectors(chunk_id: int) -> None:
    """Remove the .npz file for a chunk, if it exists."""
    path = _vector_path(chunk_id)
    if path.exists():
        path.unlink()
