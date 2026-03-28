"""SQLite + NumPy storage layer for the DAG-based embedding pipeline.

Schema: nodes + edges tables forming a directed acyclic graph.
Vectors stored as compressed .npz files keyed by node_id.
"""

import shutil
import sqlite3
from datetime import datetime, timezone
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create the nodes and edges tables if they do not exist, then migrate if needed."""
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type       TEXT NOT NULL,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                text            TEXT DEFAULT '',
                embedding_type  TEXT DEFAULT NULL,
                token_count     INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                parent_id   INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                child_id    INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                sort_order  INTEGER DEFAULT 0,
                PRIMARY KEY (parent_id, child_id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            """
        )
        conn.commit()
    finally:
        conn.close()

    _migrate_v1_to_v2()


# ---------------------------------------------------------------------------
# V1 -> V2 migration
# ---------------------------------------------------------------------------

def _has_legacy_tables() -> bool:
    conn = _connect()
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        return "documents" in tables and "chunks" in tables
    finally:
        conn.close()


def _migrate_v1_to_v2() -> None:
    """Migrate old documents+chunks tables to nodes+edges.

    For each document, creates a document node.
    For each chunk, creates a chunk node and an edge from document -> chunk.
    Vector files are copied from old chunk_id to new node_id.
    Old tables are renamed with _legacy_ prefix.
    """
    if not _has_legacy_tables():
        return

    conn = _connect()
    try:
        # Check if we already migrated (legacy tables renamed)
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "_legacy_documents" in tables:
            return

        now = _now()
        docs = conn.execute("SELECT * FROM documents").fetchall()

        for doc in docs:
            # Create document node
            cur = conn.execute(
                "INSERT INTO nodes (node_type, name, description, text, embedding_type, token_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("document", doc["filename"], "", "", None, 0, doc["ingested_at"], now),
            )
            doc_node_id = cur.lastrowid

            chunks = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc["id"],),
            ).fetchall()

            for chunk in chunks:
                # Create chunk node
                cur = conn.execute(
                    "INSERT INTO nodes (node_type, name, description, text, embedding_type, token_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "chunk",
                        f"{chunk['section']} (chunk {chunk['chunk_index']})",
                        "",
                        chunk["text"],
                        "multi",
                        chunk["token_count"],
                        doc["ingested_at"],
                        now,
                    ),
                )
                chunk_node_id = cur.lastrowid

                # Create edge
                conn.execute(
                    "INSERT INTO edges (parent_id, child_id, sort_order) VALUES (?, ?, ?)",
                    (doc_node_id, chunk_node_id, chunk["chunk_index"]),
                )

                # Copy vector file from old id to new id
                old_path = Path(VECTORS_DIR) / f"{chunk['id']}.npz"
                new_path = Path(VECTORS_DIR) / f"{chunk_node_id}.npz"
                if old_path.exists() and not new_path.exists():
                    shutil.copy2(str(old_path), str(new_path))

        # Rename old tables
        conn.execute("ALTER TABLE documents RENAME TO _legacy_documents")
        conn.execute("ALTER TABLE chunks RENAME TO _legacy_chunks")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

def insert_node(
    node_type: str,
    name: str,
    description: str = "",
    text: str = "",
    embedding_type: str | None = None,
    token_count: int = 0,
) -> int:
    """Insert a node and return its id."""
    now = _now()
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO nodes (node_type, name, description, text, embedding_type, token_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node_type, name, description, text, embedding_type, token_count, now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_node(node_id: int) -> dict | None:
    """Return a node as a dict, or None if not found."""
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_node(node_id: int, **fields) -> None:
    """Update specific fields on a node. Automatically sets updated_at."""
    if not fields:
        return
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [node_id]
    conn = _connect()
    try:
        conn.execute(f"UPDATE nodes SET {set_clause} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


def delete_node(node_id: int) -> None:
    """Delete a node, its edges, its vectors, and any orphaned descendants."""
    # Collect all descendant node ids first
    descendant_ids = [n["id"] for n in get_descendants(node_id)]
    all_ids = [node_id] + descendant_ids

    # Delete vectors for all affected nodes
    for nid in all_ids:
        delete_vectors(nid)

    conn = _connect()
    try:
        # Delete edges involving these nodes
        placeholders = ",".join("?" * len(all_ids))
        conn.execute(
            f"DELETE FROM edges WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            all_ids + all_ids,
        )
        # Delete the nodes themselves
        conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", all_ids)
        conn.commit()
    finally:
        conn.close()


def get_all_nodes(node_type: str | None = None) -> list[dict]:
    """Return all nodes, optionally filtered by type."""
    conn = _connect()
    try:
        if node_type:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE node_type = ? ORDER BY id", (node_type,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

def insert_edge(parent_id: int, child_id: int, sort_order: int = 0) -> None:
    """Create a parent -> child edge."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO edges (parent_id, child_id, sort_order) VALUES (?, ?, ?)",
            (parent_id, child_id, sort_order),
        )
        conn.commit()
    finally:
        conn.close()


def delete_edge(parent_id: int, child_id: int) -> None:
    """Remove a specific edge."""
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM edges WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_children(parent_id: int) -> list[dict]:
    """Return child nodes ordered by sort_order."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT n.* FROM nodes n
            JOIN edges e ON e.child_id = n.id
            WHERE e.parent_id = ?
            ORDER BY e.sort_order, n.id
            """,
            (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_parents(child_id: int) -> list[dict]:
    """Return parent nodes of a given node."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT n.* FROM nodes n
            JOIN edges e ON e.parent_id = n.id
            WHERE e.child_id = ?
            ORDER BY n.id
            """,
            (child_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_roots() -> list[dict]:
    """Return all nodes that have no parents (root nodes)."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT n.* FROM nodes n
            WHERE n.id NOT IN (SELECT child_id FROM edges)
            ORDER BY n.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

def get_descendants(node_id: int) -> list[dict]:
    """Return all descendant nodes (recursive BFS)."""
    visited = set()
    queue = [node_id]
    descendants = []

    while queue:
        current = queue.pop(0)
        children = get_children(current)
        for child in children:
            if child["id"] not in visited:
                visited.add(child["id"])
                descendants.append(child)
                queue.append(child["id"])

    return descendants


def get_ancestors(node_id: int) -> list[dict]:
    """Return all ancestor nodes (recursive upward BFS)."""
    visited = set()
    queue = [node_id]
    ancestors = []

    while queue:
        current = queue.pop(0)
        parents = get_parents(current)
        for parent in parents:
            if parent["id"] not in visited:
                visited.add(parent["id"])
                ancestors.append(parent)
                queue.append(parent["id"])

    return ancestors


def get_leaf_nodes(parent_id: int | None = None) -> list[dict]:
    """Return leaf nodes (nodes with no children).

    If parent_id is given, only return leaves that are descendants of that node.
    """
    if parent_id is not None:
        descendants = get_descendants(parent_id)
        return [d for d in descendants if not get_children(d["id"])]

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT n.* FROM nodes n
            WHERE n.id NOT IN (SELECT parent_id FROM edges)
            ORDER BY n.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_nodes_by_embedding_type(embedding_type: str) -> list[dict]:
    """Return all nodes with a specific embedding type."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE embedding_type = ? ORDER BY id",
            (embedding_type,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def find_node_by_source_url(url: str) -> dict | None:
    """Find a document node whose description starts with 'Source: {url}'.

    Used for fast URL-based dedup before the more expensive embedding dedup.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM nodes WHERE node_type = 'document' AND description LIKE ?",
            (f"Source: {url}%",),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def count_children(node_id: int) -> int:
    """Return the number of children for a node."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM edges WHERE parent_id = ?", (node_id,)
        ).fetchone()
        return row["cnt"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Vector file I/O
# ---------------------------------------------------------------------------

def _vector_path(node_id: int) -> Path:
    return Path(VECTORS_DIR) / f"{node_id}.npz"


def save_vectors(node_id: int, vectors: np.ndarray) -> None:
    """Save a NumPy array to data/vectors/{node_id}.npz."""
    Path(VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(_vector_path(node_id)), vectors=vectors)


def load_vectors(node_id: int) -> np.ndarray:
    """Load a NumPy array from data/vectors/{node_id}.npz."""
    data = np.load(str(_vector_path(node_id)))
    return data["vectors"]


def delete_vectors(node_id: int) -> None:
    """Remove the .npz file for a node, if it exists."""
    path = _vector_path(node_id)
    if path.exists():
        path.unlink()
