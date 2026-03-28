"""Unit tests for core.graph — graph traversal utilities."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_config(tmp_path, monkeypatch):
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
# get_graph_description
# ---------------------------------------------------------------------------

class TestGetGraphDescription:

    def test_empty_graph(self, store):
        from core.graph import get_graph_description
        desc = get_graph_description()
        assert desc == "(empty graph)"

    def test_single_root(self, store):
        from core.graph import get_graph_description
        store.insert_node("category", "Finance")
        desc = get_graph_description()
        assert "Finance" in desc

    def test_nested_structure(self, store):
        from core.graph import get_graph_description
        root = store.insert_node("category", "News")
        child = store.insert_node("document", "Article 1")
        store.insert_edge(root, child)

        desc = get_graph_description()
        assert "News" in desc
        assert "Article 1" in desc

    def test_shows_child_counts(self, store):
        from core.graph import get_graph_description
        root = store.insert_node("category", "News")
        store.insert_node("document", "A1")
        store.insert_node("document", "A2")
        c1 = store.insert_node("document", "A1")
        c2 = store.insert_node("document", "A2")
        store.insert_edge(root, c1)
        store.insert_edge(root, c2)

        desc = get_graph_description()
        assert "2 children" in desc

    def test_respects_max_depth(self, store):
        from core.graph import get_graph_description
        root = store.insert_node("category", "Root")
        mid = store.insert_node("document", "Mid")
        leaf = store.insert_node("chunk", "DeepLeaf")
        store.insert_edge(root, mid)
        store.insert_edge(mid, leaf)

        # depth 1 should show Root and Mid, but not DeepLeaf
        desc = get_graph_description(max_depth=1)
        assert "Root" in desc
        assert "Mid" in desc
        assert "DeepLeaf" not in desc

    def test_shows_description_preview(self, store):
        from core.graph import get_graph_description
        store.insert_node("category", "Finance", description="Financial markets and economic data")
        desc = get_graph_description()
        assert "Financial markets" in desc


# ---------------------------------------------------------------------------
# find_leaf_descendants
# ---------------------------------------------------------------------------

class TestFindLeafDescendants:

    def test_returns_leaf_ids(self, store):
        from core.graph import find_leaf_descendants
        root = store.insert_node("category", "Root")
        doc = store.insert_node("document", "Doc")
        leaf1 = store.insert_node("chunk", "L1")
        leaf2 = store.insert_node("chunk", "L2")
        store.insert_edge(root, doc)
        store.insert_edge(doc, leaf1)
        store.insert_edge(doc, leaf2)

        leaves = find_leaf_descendants(root)
        assert set(leaves) == {leaf1, leaf2}

    def test_excludes_branch_nodes(self, store):
        from core.graph import find_leaf_descendants
        root = store.insert_node("category", "Root")
        doc = store.insert_node("document", "Doc")
        leaf = store.insert_node("chunk", "Leaf")
        store.insert_edge(root, doc)
        store.insert_edge(doc, leaf)

        leaves = find_leaf_descendants(root)
        assert doc not in leaves

    def test_empty_when_no_leaves(self, store):
        from core.graph import find_leaf_descendants
        # A node with no children IS a leaf, but find_leaf_descendants
        # only looks at descendants, not the node itself
        root = store.insert_node("category", "Root")
        leaves = find_leaf_descendants(root)
        assert leaves == []


# ---------------------------------------------------------------------------
# compute_depth
# ---------------------------------------------------------------------------

class TestComputeDepth:

    def test_root_depth_zero(self, store):
        from core.graph import compute_depth
        root = store.insert_node("category", "Root")
        assert compute_depth(root) == 0

    def test_child_depth_one(self, store):
        from core.graph import compute_depth
        root = store.insert_node("category", "Root")
        child = store.insert_node("document", "Child")
        store.insert_edge(root, child)
        assert compute_depth(child) == 1

    def test_grandchild_depth_two(self, store):
        from core.graph import compute_depth
        root = store.insert_node("category", "Root")
        mid = store.insert_node("document", "Mid")
        leaf = store.insert_node("chunk", "Leaf")
        store.insert_edge(root, mid)
        store.insert_edge(mid, leaf)
        assert compute_depth(leaf) == 2

    def test_multiple_parents_uses_shortest_path(self, store):
        from core.graph import compute_depth
        root = store.insert_node("category", "Root")
        mid = store.insert_node("document", "Mid")
        child = store.insert_node("chunk", "Child")
        store.insert_edge(root, mid)
        store.insert_edge(mid, child)
        store.insert_edge(root, child)  # shortcut: root -> child directly

        # Child is reachable at depth 1 (root->child) and depth 2 (root->mid->child)
        # Should return the shortest: 1
        assert compute_depth(child) == 1
