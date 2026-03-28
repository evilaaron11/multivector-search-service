"""Unit tests for the service orchestration layer.

All underlying modules (store, chunker, embedder, searcher, llm) are mocked
so no real DB, API, filesystem, or CLI access occurs.
"""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest

import service


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_CHUNKS = [
    {"section": "Intro", "text": "Hello world", "chunk_index": 0, "token_count": 10},
    {"section": "Body", "text": "More content", "chunk_index": 0, "token_count": 12},
]

FAKE_EMBEDDINGS_MULTI = [
    np.random.randn(10, 128).astype(np.float32),
    np.random.randn(12, 128).astype(np.float32),
]

FAKE_SUMMARY_VECTOR = np.random.randn(1024).astype(np.float32)

FAKE_SEARCH_RESULTS = [
    {
        "node_id": 1,
        "score": 5.2,
        "text": "result text",
        "name": "chunk 0",
        "node_type": "chunk",
        "highlights": [("result", 0.85, "query")],
    },
]


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestDocument:

    @patch("service.update_ancestor_summaries")
    @patch("service.embedder")
    @patch("service.store")
    @patch("service.chunker")
    @patch("service.generate_leaf_description", return_value="A document about testing")
    @patch("service.confirm_duplicate", return_value={"is_duplicate": False})
    @patch("service.propose_placement", return_value={
        "action": "create_new_branch",
        "target_node_id": None,
        "new_branch_name": "Uncategorized",
        "new_branch_description": "",
        "confidence": 0.5,
        "reasoning": "No existing nodes",
    })
    def test_ingest_creates_document_and_chunks(
        self, mock_placement, mock_dedup, mock_desc, mock_chunker, mock_store, mock_embedder, mock_update
    ):
        mock_chunker.process_markdown.return_value = FAKE_CHUNKS
        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = []
        mock_store.insert_node.side_effect = [100, 101, 102]  # doc, chunk1, chunk2
        mock_store.save_vectors.return_value = None
        mock_store.insert_edge.return_value = None
        mock_store.get_node.return_value = None
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]
        mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS_MULTI

        result = service.ingest_document("test.md")

        assert result.node_id == 100
        assert result.name == "test.md"
        assert result.num_chunks == 2
        assert "Intro" in result.sections
        assert "Body" in result.sections

    @patch("service.update_ancestor_summaries")
    @patch("service.embedder")
    @patch("service.store")
    @patch("service.chunker")
    @patch("service.generate_leaf_description", return_value="A document")
    @patch("service.confirm_duplicate", return_value={"is_duplicate": True})
    def test_ingest_skips_duplicate(
        self, mock_dedup, mock_desc, mock_chunker, mock_store, mock_embedder, mock_update
    ):
        mock_chunker.process_markdown.return_value = FAKE_CHUNKS
        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [
            {"id": 1, "node_type": "document", "embedding_type": "single", "name": "existing", "description": "existing"}
        ]
        mock_store.load_vectors.return_value = FAKE_SUMMARY_VECTOR
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]

        result = service.ingest_document("duplicate.md")

        assert result.node_id == -1
        assert result.num_chunks == 0

    @patch("service.update_ancestor_summaries")
    @patch("service.embedder")
    @patch("service.store")
    @patch("service.chunker")
    @patch("service.generate_leaf_description", return_value="doc summary")
    @patch("service.confirm_duplicate", return_value={"is_duplicate": False})
    @patch("service.propose_placement", return_value={
        "action": "attach_to_existing",
        "target_node_id": 50,
        "confidence": 0.9,
        "reasoning": "Matches Finance",
    })
    def test_ingest_attaches_to_existing_parent(
        self, mock_placement, mock_dedup, mock_desc, mock_chunker, mock_store, mock_embedder, mock_update
    ):
        mock_chunker.process_markdown.return_value = FAKE_CHUNKS
        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [
            {"id": 50, "name": "Finance", "node_type": "category",
             "embedding_type": "single", "description": "Financial news"}
        ]
        mock_store.load_vectors.return_value = FAKE_SUMMARY_VECTOR
        mock_store.insert_node.side_effect = [200, 201, 202]
        mock_store.save_vectors.return_value = None
        mock_store.insert_edge.return_value = None
        mock_store.get_node.return_value = {"node_type": "category", "name": "Finance"}
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]
        mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS_MULTI

        result = service.ingest_document("finance_article.md")

        assert result.parent_node_id == 50
        mock_store.insert_edge.assert_any_call(50, 200)

    @patch("service.update_ancestor_summaries")
    @patch("service.embedder")
    @patch("service.store")
    @patch("service.chunker")
    @patch("service.generate_leaf_description", return_value="doc summary")
    @patch("service.confirm_duplicate", return_value={"is_duplicate": False})
    @patch("service.propose_placement", return_value={
        "action": "attach_to_existing",
        "target_node_id": 50,
        "confidence": 0.5,
        "reasoning": "Uncertain match",
    })
    def test_interactive_mode_calls_callback(
        self, mock_placement, mock_dedup, mock_desc, mock_chunker, mock_store, mock_embedder, mock_update
    ):
        mock_chunker.process_markdown.return_value = FAKE_CHUNKS
        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [
            {"id": 50, "name": "Finance", "node_type": "category",
             "embedding_type": "single", "description": "Financial news"}
        ]
        mock_store.load_vectors.return_value = FAKE_SUMMARY_VECTOR
        mock_store.insert_node.side_effect = [200, 201, 202]
        mock_store.save_vectors.return_value = None
        mock_store.insert_edge.return_value = None
        mock_store.get_node.return_value = {"node_type": "category", "name": "Finance"}
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]
        mock_embedder.embed_documents.return_value = FAKE_EMBEDDINGS_MULTI

        callback = MagicMock(return_value=0)  # pick first option
        result = service.ingest_document("uncertain.md", confirm_callback=callback)

        callback.assert_called_once()
        assert result.node_id == 200


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearchDocuments:

    @patch("service.searcher")
    @patch("service.store")
    def test_flat_search(self, mock_store, mock_searcher):
        mock_store.init_db.return_value = None
        mock_searcher.flat_search.return_value = FAKE_SEARCH_RESULTS

        results = service.search_documents("test query", flat=True)

        mock_searcher.flat_search.assert_called_once_with("test query", top_k=5, node_id=None)
        assert len(results) == 1
        assert isinstance(results[0], service.SearchResult)
        assert results[0].score == 5.2

    @patch("service.searcher")
    @patch("service.store")
    def test_graph_search(self, mock_store, mock_searcher):
        mock_store.init_db.return_value = None
        mock_searcher.graph_search.return_value = FAKE_SEARCH_RESULTS

        results = service.search_documents("test query", flat=False)

        mock_searcher.graph_search.assert_called_once_with("test query", top_k=5)
        assert len(results) == 1

    @patch("service.searcher")
    @patch("service.store")
    def test_respects_top_k(self, mock_store, mock_searcher):
        mock_store.init_db.return_value = None
        mock_searcher.graph_search.return_value = []

        service.search_documents("query", top_k=10)

        mock_searcher.graph_search.assert_called_once_with("query", top_k=10)


# ---------------------------------------------------------------------------
# List and delete tests
# ---------------------------------------------------------------------------

class TestListNodes:

    @patch("service.store")
    def test_returns_node_info(self, mock_store):
        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [
            {"id": 1, "name": "News", "node_type": "category",
             "description": "News articles", "embedding_type": "single"},
            {"id": 2, "name": "Chunk 0", "node_type": "chunk",
             "description": "", "embedding_type": "multi"},
        ]
        mock_store.count_children.side_effect = [3, 0]

        results = service.list_nodes()

        assert len(results) == 2
        assert isinstance(results[0], service.NodeInfo)
        assert results[0].name == "News"
        assert results[0].child_count == 3
        assert results[1].child_count == 0


class TestDeleteNode:

    @patch("service.store")
    def test_delete_existing(self, mock_store):
        mock_store.init_db.return_value = None
        mock_store.get_node.return_value = {"id": 1, "name": "test"}
        mock_store.delete_node.return_value = None

        assert service.delete_node(1) is True
        mock_store.delete_node.assert_called_once_with(1)

    @patch("service.store")
    def test_delete_nonexistent(self, mock_store):
        mock_store.init_db.return_value = None
        mock_store.get_node.return_value = None

        assert service.delete_node(999) is False


# ---------------------------------------------------------------------------
# Enrichment tests
# ---------------------------------------------------------------------------

class TestEnrichGraph:

    @patch("service.embedder")
    @patch("service.store")
    @patch("service.generate_node_summary", return_value="Branch summary")
    @patch("service.generate_leaf_description", return_value="Leaf description")
    def test_enriches_leaves_and_branches(
        self, mock_leaf_desc, mock_branch_desc, mock_store, mock_embedder
    ):
        leaf = {"id": 1, "name": "Chunk", "node_type": "chunk",
                "text": "content", "description": "", "embedding_type": "multi"}
        branch = {"id": 2, "name": "Doc", "node_type": "document",
                  "text": "", "description": "", "embedding_type": None}

        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [leaf, branch]
        mock_store.get_children.side_effect = lambda nid: [leaf] if nid == 2 else []
        mock_store.get_ancestors.return_value = []
        mock_store.update_node.return_value = None
        mock_store.save_vectors.return_value = None
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]

        count = service.enrich_graph()

        assert count == 2
        mock_leaf_desc.assert_called_once()
        mock_branch_desc.assert_called_once()

    @patch("service.store")
    def test_skips_nodes_with_descriptions(self, mock_store):
        leaf = {"id": 1, "name": "C", "node_type": "chunk",
                "text": "content", "description": "already described", "embedding_type": "multi"}

        mock_store.init_db.return_value = None
        mock_store.get_all_nodes.return_value = [leaf]
        mock_store.get_children.return_value = []

        count = service.enrich_graph()
        assert count == 0


# ---------------------------------------------------------------------------
# Summary propagation tests
# ---------------------------------------------------------------------------

class TestUpdateAncestorSummaries:

    @patch("service.embedder")
    @patch("service.store")
    @patch("service.generate_node_summary", return_value="Updated summary")
    def test_updates_ancestors(self, mock_summary, mock_store, mock_embedder):
        mock_store.get_ancestors.return_value = [
            {"id": 10, "name": "Parent", "node_type": "category"}
        ]
        mock_store.get_children.return_value = [
            {"id": 20, "name": "Child", "description": "child desc"}
        ]
        mock_store.update_node.return_value = None
        mock_store.save_vectors.return_value = None
        mock_embedder.embed_single_vector.return_value = [FAKE_SUMMARY_VECTOR]

        service.update_ancestor_summaries(20)

        mock_summary.assert_called_once()
        mock_store.update_node.assert_any_call(10, description="Updated summary")
        mock_store.save_vectors.assert_called()
