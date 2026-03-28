"""Unit tests for core.store — nodes + edges DAG storage layer."""

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

    import importlib
    import core.store as store_mod
    importlib.reload(store_mod)

    monkeypatch.setattr(store_mod, "DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "VECTORS_DIR", vectors_dir)


@pytest.fixture
def store():
    import core.store as mod
    mod.init_db()
    return mod


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(store):
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
    assert "nodes" in tables
    assert "edges" in tables


def test_init_db_is_idempotent(store):
    store.init_db()


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

def test_insert_and_get_node(store):
    nid = store.insert_node("document", "report.pdf", description="A report")
    assert isinstance(nid, int)

    node = store.get_node(nid)
    assert node is not None
    assert node["name"] == "report.pdf"
    assert node["node_type"] == "document"
    assert node["description"] == "A report"
    assert "created_at" in node
    assert "updated_at" in node


def test_get_node_nonexistent(store):
    assert store.get_node(9999) is None


def test_insert_node_defaults(store):
    nid = store.insert_node("chunk", "chunk 0")
    node = store.get_node(nid)
    assert node["description"] == ""
    assert node["text"] == ""
    assert node["embedding_type"] is None
    assert node["token_count"] == 0


def test_insert_node_with_all_fields(store):
    nid = store.insert_node(
        node_type="chunk",
        name="Section 1 (chunk 0)",
        description="First chunk",
        text="Hello world",
        embedding_type="multi",
        token_count=42,
    )
    node = store.get_node(nid)
    assert node["text"] == "Hello world"
    assert node["embedding_type"] == "multi"
    assert node["token_count"] == 42


def test_update_node(store):
    nid = store.insert_node("document", "old_name")
    store.update_node(nid, name="new_name", description="updated desc")
    node = store.get_node(nid)
    assert node["name"] == "new_name"
    assert node["description"] == "updated desc"


def test_update_node_sets_updated_at(store):
    nid = store.insert_node("document", "doc")
    node_before = store.get_node(nid)
    store.update_node(nid, description="changed")
    node_after = store.get_node(nid)
    assert node_after["updated_at"] >= node_before["updated_at"]


def test_update_node_noop_with_no_fields(store):
    nid = store.insert_node("document", "doc")
    store.update_node(nid)  # should not raise


def test_delete_node_simple(store):
    nid = store.insert_node("document", "bye")
    store.delete_node(nid)
    assert store.get_node(nid) is None


def test_delete_node_cascades_to_children(store):
    parent = store.insert_node("category", "News")
    child1 = store.insert_node("document", "Article 1")
    child2 = store.insert_node("document", "Article 2")
    store.insert_edge(parent, child1)
    store.insert_edge(parent, child2)

    # Save vectors for children
    arr = np.random.rand(4, 128).astype(np.float32)
    store.save_vectors(child1, arr)
    store.save_vectors(child2, arr)

    store.delete_node(parent)

    assert store.get_node(parent) is None
    assert store.get_node(child1) is None
    assert store.get_node(child2) is None

    from config import VECTORS_DIR
    assert not (VECTORS_DIR / f"{child1}.npz").exists()
    assert not (VECTORS_DIR / f"{child2}.npz").exists()


def test_delete_node_cascades_deep(store):
    """Deleting a root should remove all descendants."""
    root = store.insert_node("category", "Root")
    mid = store.insert_node("document", "Mid")
    leaf = store.insert_node("chunk", "Leaf")
    store.insert_edge(root, mid)
    store.insert_edge(mid, leaf)

    store.delete_node(root)

    assert store.get_node(root) is None
    assert store.get_node(mid) is None
    assert store.get_node(leaf) is None


def test_get_all_nodes(store):
    store.insert_node("document", "a")
    store.insert_node("chunk", "b")
    store.insert_node("document", "c")

    all_nodes = store.get_all_nodes()
    assert len(all_nodes) == 3

    docs = store.get_all_nodes(node_type="document")
    assert len(docs) == 2

    chunks = store.get_all_nodes(node_type="chunk")
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

def test_insert_and_get_children(store):
    parent = store.insert_node("category", "Parent")
    child1 = store.insert_node("document", "Child1")
    child2 = store.insert_node("document", "Child2")

    store.insert_edge(parent, child1, sort_order=0)
    store.insert_edge(parent, child2, sort_order=1)

    children = store.get_children(parent)
    assert len(children) == 2
    assert children[0]["name"] == "Child1"
    assert children[1]["name"] == "Child2"


def test_get_children_ordered(store):
    parent = store.insert_node("category", "Parent")
    c1 = store.insert_node("chunk", "C")
    c2 = store.insert_node("chunk", "A")
    c3 = store.insert_node("chunk", "B")

    store.insert_edge(parent, c1, sort_order=2)
    store.insert_edge(parent, c2, sort_order=0)
    store.insert_edge(parent, c3, sort_order=1)

    children = store.get_children(parent)
    assert [c["name"] for c in children] == ["A", "B", "C"]


def test_get_children_empty(store):
    nid = store.insert_node("document", "Lonely")
    assert store.get_children(nid) == []


def test_get_parents(store):
    parent1 = store.insert_node("category", "Tech")
    parent2 = store.insert_node("category", "Finance")
    child = store.insert_node("document", "Apple Earnings")

    store.insert_edge(parent1, child)
    store.insert_edge(parent2, child)

    parents = store.get_parents(child)
    assert len(parents) == 2
    names = {p["name"] for p in parents}
    assert names == {"Tech", "Finance"}


def test_get_parents_empty(store):
    nid = store.insert_node("category", "Root")
    assert store.get_parents(nid) == []


def test_delete_edge(store):
    parent = store.insert_node("category", "Parent")
    child = store.insert_node("document", "Child")
    store.insert_edge(parent, child)

    assert len(store.get_children(parent)) == 1
    store.delete_edge(parent, child)
    assert len(store.get_children(parent)) == 0


def test_insert_edge_duplicate_ignored(store):
    parent = store.insert_node("category", "P")
    child = store.insert_node("document", "C")
    store.insert_edge(parent, child)
    store.insert_edge(parent, child)  # should not raise
    assert len(store.get_children(parent)) == 1


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

def test_get_roots(store):
    root1 = store.insert_node("category", "News")
    root2 = store.insert_node("category", "Research")
    child = store.insert_node("document", "Article")
    store.insert_edge(root1, child)

    roots = store.get_roots()
    root_ids = {r["id"] for r in roots}
    assert root1 in root_ids
    assert root2 in root_ids
    assert child not in root_ids


def test_get_descendants(store):
    root = store.insert_node("category", "Root")
    mid = store.insert_node("document", "Mid")
    leaf1 = store.insert_node("chunk", "Leaf1")
    leaf2 = store.insert_node("chunk", "Leaf2")

    store.insert_edge(root, mid)
    store.insert_edge(mid, leaf1)
    store.insert_edge(mid, leaf2)

    descendants = store.get_descendants(root)
    desc_ids = {d["id"] for d in descendants}
    assert desc_ids == {mid, leaf1, leaf2}


def test_get_descendants_empty(store):
    nid = store.insert_node("chunk", "Leaf")
    assert store.get_descendants(nid) == []


def test_get_ancestors(store):
    root = store.insert_node("category", "Root")
    mid = store.insert_node("document", "Mid")
    leaf = store.insert_node("chunk", "Leaf")

    store.insert_edge(root, mid)
    store.insert_edge(mid, leaf)

    ancestors = store.get_ancestors(leaf)
    anc_ids = {a["id"] for a in ancestors}
    assert anc_ids == {root, mid}


def test_get_ancestors_empty(store):
    nid = store.insert_node("category", "Root")
    assert store.get_ancestors(nid) == []


def test_get_leaf_nodes_global(store):
    root = store.insert_node("category", "Root")
    doc = store.insert_node("document", "Doc")
    leaf1 = store.insert_node("chunk", "Leaf1")
    leaf2 = store.insert_node("chunk", "Leaf2")

    store.insert_edge(root, doc)
    store.insert_edge(doc, leaf1)
    store.insert_edge(doc, leaf2)

    leaves = store.get_leaf_nodes()
    leaf_ids = {l["id"] for l in leaves}
    assert leaf1 in leaf_ids
    assert leaf2 in leaf_ids
    assert doc not in leaf_ids
    assert root not in leaf_ids


def test_get_leaf_nodes_under_parent(store):
    root = store.insert_node("category", "Root")
    doc1 = store.insert_node("document", "Doc1")
    doc2 = store.insert_node("document", "Doc2")
    leaf1 = store.insert_node("chunk", "Leaf1")
    leaf2 = store.insert_node("chunk", "Leaf2")

    store.insert_edge(root, doc1)
    store.insert_edge(root, doc2)
    store.insert_edge(doc1, leaf1)
    store.insert_edge(doc2, leaf2)

    leaves = store.get_leaf_nodes(parent_id=doc1)
    assert len(leaves) == 1
    assert leaves[0]["id"] == leaf1


def test_get_nodes_by_embedding_type(store):
    store.insert_node("document", "A", embedding_type="single")
    store.insert_node("chunk", "B", embedding_type="multi")
    store.insert_node("chunk", "C", embedding_type="multi")

    singles = store.get_nodes_by_embedding_type("single")
    assert len(singles) == 1

    multis = store.get_nodes_by_embedding_type("multi")
    assert len(multis) == 2


def test_count_children(store):
    parent = store.insert_node("category", "P")
    c1 = store.insert_node("document", "C1")
    c2 = store.insert_node("document", "C2")
    store.insert_edge(parent, c1)
    store.insert_edge(parent, c2)

    assert store.count_children(parent) == 2
    assert store.count_children(c1) == 0


# ---------------------------------------------------------------------------
# DAG (multiple parents)
# ---------------------------------------------------------------------------

def test_dag_multiple_parents(store):
    """A node can have multiple parents (DAG, not tree)."""
    p1 = store.insert_node("category", "Tech")
    p2 = store.insert_node("category", "Finance")
    child = store.insert_node("document", "Apple Earnings")

    store.insert_edge(p1, child)
    store.insert_edge(p2, child)

    assert len(store.get_parents(child)) == 2
    assert child in {c["id"] for c in store.get_children(p1)}
    assert child in {c["id"] for c in store.get_children(p2)}


# ---------------------------------------------------------------------------
# Vector file I/O
# ---------------------------------------------------------------------------

def test_save_and_load_vectors_2d(store):
    nid = store.insert_node("chunk", "c", embedding_type="multi")
    original = np.random.rand(10, 128).astype(np.float32)
    store.save_vectors(nid, original)
    loaded = store.load_vectors(nid)
    np.testing.assert_array_equal(original, loaded)


def test_save_and_load_vectors_1d(store):
    """Single-vector nodes store 1-D arrays."""
    nid = store.insert_node("document", "d", embedding_type="single")
    original = np.random.rand(1024).astype(np.float32)
    store.save_vectors(nid, original)
    loaded = store.load_vectors(nid)
    np.testing.assert_array_equal(original, loaded)


def test_delete_vectors_removes_file(store):
    nid = store.insert_node("chunk", "c")
    store.save_vectors(nid, np.array([[1.0, 2.0]]))
    store.delete_vectors(nid)

    from config import VECTORS_DIR
    assert not (VECTORS_DIR / f"{nid}.npz").exists()


def test_delete_vectors_noop_when_missing(store):
    store.delete_vectors(99999)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migrate_v1_to_v2(tmp_path, monkeypatch):
    """Create a v1-style database and verify migration to v2."""
    import sqlite3
    import config
    import importlib

    db_path = tmp_path / "migrate_test.db"
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "VECTORS_DIR", vectors_dir)

    import core.store as store_mod
    importlib.reload(store_mod)
    monkeypatch.setattr(store_mod, "DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "VECTORS_DIR", vectors_dir)

    # Create v1 schema manually
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            text TEXT NOT NULL,
            section TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL
        );
    """)
    conn.execute("INSERT INTO documents (filename, ingested_at) VALUES ('test.md', '2024-01-01')")
    conn.execute("INSERT INTO chunks (document_id, text, section, chunk_index, token_count) VALUES (1, 'hello', 'intro', 0, 5)")
    conn.execute("INSERT INTO chunks (document_id, text, section, chunk_index, token_count) VALUES (1, 'world', 'body', 1, 5)")
    conn.commit()

    # Save vector files with old chunk IDs
    vec = np.random.rand(5, 128).astype(np.float32)
    np.savez_compressed(str(vectors_dir / "1.npz"), vectors=vec)
    np.savez_compressed(str(vectors_dir / "2.npz"), vectors=vec)

    conn.close()

    # Run init_db which triggers migration
    store_mod.init_db()

    # Verify new tables exist
    conn = sqlite3.connect(str(db_path))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "nodes" in tables
    assert "edges" in tables
    assert "_legacy_documents" in tables
    assert "_legacy_chunks" in tables
    assert "documents" not in tables
    assert "chunks" not in tables
    conn.close()

    # Verify nodes were created
    nodes = store_mod.get_all_nodes()
    doc_nodes = [n for n in nodes if n["node_type"] == "document"]
    chunk_nodes = [n for n in nodes if n["node_type"] == "chunk"]
    assert len(doc_nodes) == 1
    assert len(chunk_nodes) == 2
    assert doc_nodes[0]["name"] == "test.md"

    # Verify edges
    children = store_mod.get_children(doc_nodes[0]["id"])
    assert len(children) == 2

    # Verify chunk content preserved
    chunk_texts = {c["text"] for c in chunk_nodes}
    assert chunk_texts == {"hello", "world"}

    # Verify vector files were copied to new IDs
    for chunk_node in chunk_nodes:
        vec_path = vectors_dir / f"{chunk_node['id']}.npz"
        assert vec_path.exists()
