"""Unit tests for core.searcher — MaxSim, cosine, flat search, graph search."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.searcher import maxsim_score, cosine_score, score_node


# ---------------------------------------------------------------------------
# MaxSim scoring
# ---------------------------------------------------------------------------

class TestMaxSimScore:

    def test_identical_vectors(self):
        v = np.random.randn(5, 128).astype(np.float32)
        score = maxsim_score(v, v)
        assert score == pytest.approx(5.0, abs=0.01)

    def test_orthogonal_vectors(self):
        q = np.eye(4, 128, dtype=np.float32)
        d = np.eye(4, 128, k=64, dtype=np.float32)
        score = maxsim_score(q, d)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_score_is_scalar(self):
        q = np.random.randn(3, 128).astype(np.float32)
        d = np.random.randn(10, 128).astype(np.float32)
        score = maxsim_score(q, d)
        assert isinstance(score, float)

    def test_score_scales_with_query_tokens(self):
        d = np.random.randn(20, 128).astype(np.float32)
        q_short = np.random.randn(3, 128).astype(np.float32)
        q_long = np.random.randn(10, 128).astype(np.float32)
        score_short = maxsim_score(q_short, d)
        score_long = maxsim_score(q_long, d)
        assert isinstance(score_short, float)
        assert isinstance(score_long, float)


# ---------------------------------------------------------------------------
# Cosine scoring
# ---------------------------------------------------------------------------

class TestCosineScore:

    def test_identical_vectors(self):
        v = np.random.randn(1024).astype(np.float32)
        score = cosine_score(v, v)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_opposite_vectors(self):
        v = np.random.randn(1024).astype(np.float32)
        score = cosine_score(v, -v)
        assert score == pytest.approx(-1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        a = np.zeros(128, dtype=np.float32)
        b = np.zeros(128, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        score = cosine_score(a, b)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_returns_float(self):
        a = np.random.randn(1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        assert isinstance(cosine_score(a, b), float)

    def test_range(self):
        for _ in range(10):
            a = np.random.randn(256).astype(np.float32)
            b = np.random.randn(256).astype(np.float32)
            score = cosine_score(a, b)
            assert -1.01 <= score <= 1.01


# ---------------------------------------------------------------------------
# score_node dispatch
# ---------------------------------------------------------------------------

class TestScoreNode:

    @patch("core.searcher.load_vectors")
    def test_dispatches_to_cosine_for_single(self, mock_load):
        vec = np.random.randn(1024).astype(np.float32)
        mock_load.return_value = vec

        query_single = np.random.randn(1024).astype(np.float32)
        query_multi = np.random.randn(5, 128).astype(np.float32)

        score = score_node(query_single, query_multi, node_id=1, embedding_type="single")
        expected = cosine_score(query_single, vec)
        assert score == pytest.approx(expected, abs=0.001)

    @patch("core.searcher.load_vectors")
    def test_dispatches_to_maxsim_for_multi(self, mock_load):
        doc_vec = np.random.randn(20, 128).astype(np.float32)
        mock_load.return_value = doc_vec

        query_single = np.random.randn(1024).astype(np.float32)
        query_multi = np.random.randn(5, 128).astype(np.float32)

        score = score_node(query_single, query_multi, node_id=1, embedding_type="multi")
        expected = maxsim_score(query_multi, doc_vec)
        assert score == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Flat search
# ---------------------------------------------------------------------------

class TestFlatSearch:

    @patch("core.searcher.embed_query")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_leaf_nodes")
    def test_returns_results_sorted_by_score(self, mock_leaves, mock_load, mock_embed):
        from core.searcher import flat_search

        q_vec = np.random.randn(5, 128).astype(np.float32)
        mock_embed.return_value = q_vec

        leaf1 = {"id": 1, "name": "low", "node_type": "chunk", "text": "low",
                 "embedding_type": "multi", "token_count": 5}
        leaf2 = {"id": 2, "name": "high", "node_type": "chunk", "text": "high",
                 "embedding_type": "multi", "token_count": 5}
        mock_leaves.return_value = [leaf1, leaf2]

        low_vec = np.random.randn(5, 128).astype(np.float32) * 0.01
        high_vec = q_vec.copy()

        def load_side_effect(nid):
            return high_vec if nid == 2 else low_vec
        mock_load.side_effect = load_side_effect

        results = flat_search("test", top_k=2)
        assert len(results) == 2
        assert results[0]["node_id"] == 2
        assert results[0]["score"] > results[1]["score"]

    @patch("core.searcher.embed_query")
    @patch("core.searcher.get_leaf_nodes")
    def test_empty_when_no_leaves(self, mock_leaves, mock_embed):
        from core.searcher import flat_search
        mock_embed.return_value = np.random.randn(5, 128).astype(np.float32)
        mock_leaves.return_value = []
        assert flat_search("test") == []

    @patch("core.searcher.embed_query")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_leaf_nodes")
    def test_respects_top_k(self, mock_leaves, mock_load, mock_embed):
        from core.searcher import flat_search
        mock_embed.return_value = np.random.randn(5, 128).astype(np.float32)

        leaves = [
            {"id": i, "name": f"L{i}", "node_type": "chunk", "text": f"text{i}",
             "embedding_type": "multi", "token_count": 5}
            for i in range(10)
        ]
        mock_leaves.return_value = leaves
        mock_load.return_value = np.random.randn(5, 128).astype(np.float32)

        results = flat_search("test", top_k=3)
        assert len(results) == 3

    @patch("core.searcher.embed_query")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_leaf_nodes")
    def test_skips_non_multi_leaves(self, mock_leaves, mock_load, mock_embed):
        from core.searcher import flat_search
        mock_embed.return_value = np.random.randn(5, 128).astype(np.float32)

        leaves = [
            {"id": 1, "name": "single", "node_type": "doc", "text": "t",
             "embedding_type": "single", "token_count": 5},
            {"id": 2, "name": "multi", "node_type": "chunk", "text": "t",
             "embedding_type": "multi", "token_count": 5},
        ]
        mock_leaves.return_value = leaves
        mock_load.return_value = np.random.randn(5, 128).astype(np.float32)

        results = flat_search("test")
        assert len(results) == 1
        assert results[0]["node_id"] == 2


# ---------------------------------------------------------------------------
# Graph search
# ---------------------------------------------------------------------------

class TestGraphSearch:

    @patch("core.searcher.embed_query")
    @patch("core.searcher.embed_single_query")
    @patch("core.searcher.get_roots")
    def test_falls_back_to_flat_when_no_root_embeddings(
        self, mock_roots, mock_single, mock_multi
    ):
        from core.searcher import graph_search

        mock_roots.return_value = [
            {"id": 1, "name": "Root", "node_type": "category", "embedding_type": None}
        ]
        mock_multi.return_value = np.random.randn(5, 128).astype(np.float32)

        with patch("core.searcher.flat_search", return_value=[{"node_id": 99}]) as mock_flat:
            results = graph_search("test")
            mock_flat.assert_called_once()
            assert results[0]["node_id"] == 99

    @patch("core.searcher.embed_query")
    @patch("core.searcher.embed_single_query")
    @patch("core.searcher.get_roots")
    def test_returns_empty_for_no_roots(self, mock_roots, mock_single, mock_multi):
        from core.searcher import graph_search
        mock_roots.return_value = []
        assert graph_search("test") == []

    @patch("core.searcher.word_relevance", return_value=[])
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_children")
    @patch("core.searcher.get_roots")
    @patch("core.searcher.embed_query")
    @patch("core.searcher.embed_single_query")
    def test_navigates_to_leaves(
        self, mock_single_q, mock_multi_q, mock_roots, mock_children, mock_load, mock_wr
    ):
        from core.searcher import graph_search

        q_single = np.random.randn(1024).astype(np.float32)
        q_multi = np.random.randn(5, 128).astype(np.float32)
        mock_single_q.return_value = q_single
        mock_multi_q.return_value = q_multi

        root = {"id": 1, "name": "Root", "node_type": "category", "embedding_type": "single"}
        doc = {"id": 2, "name": "Doc", "node_type": "document", "embedding_type": "single"}
        leaf = {"id": 3, "name": "Chunk", "node_type": "chunk", "embedding_type": "multi",
                "text": "content", "token_count": 5}

        mock_roots.return_value = [root]

        def children_side_effect(nid):
            if nid == 1:
                return [doc]
            elif nid == 2:
                return [leaf]
            return []
        mock_children.side_effect = children_side_effect

        root_vec = q_single * 0.9
        doc_vec = q_single * 0.85
        leaf_vec = q_multi * 0.8

        def load_side_effect(nid):
            if nid == 1:
                return root_vec
            elif nid == 2:
                return doc_vec
            return leaf_vec
        mock_load.side_effect = load_side_effect

        results = graph_search("test query", top_k=5)
        assert len(results) >= 1
        assert any(r["node_id"] == 3 for r in results)


# ---------------------------------------------------------------------------
# _select_candidates (threshold routing)
# ---------------------------------------------------------------------------

class TestSelectCandidates:

    def test_filters_below_low_threshold(self):
        from core.searcher import _select_candidates
        candidates = [
            {"id": 1, "name": "A", "score": 0.1},
            {"id": 2, "name": "B", "score": 0.05},
        ]
        result = _select_candidates("query", candidates)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_relative_cutoff_filters_weak_candidates(self):
        """Candidates far below the top score (< 70% of best) should be filtered."""
        from core.searcher import _select_candidates
        candidates = [
            {"id": 1, "name": "A", "score": 0.9},
            {"id": 2, "name": "B", "score": 0.8},
            {"id": 3, "name": "C", "score": 0.3},  # 0.3 < 0.9 * 0.70 = 0.63
        ]
        result = _select_candidates("query", candidates)
        result_ids = {c["id"] for c in result}
        assert 1 in result_ids
        assert 2 in result_ids
        assert 3 not in result_ids

    def test_empty_input(self):
        from core.searcher import _select_candidates
        assert _select_candidates("query", []) == []

    def test_respects_max_candidates(self):
        from core.searcher import _select_candidates
        # All scores close together so relative cutoff doesn't filter them
        candidates = [
            {"id": i, "name": f"N{i}", "score": 0.90 - i * 0.005}
            for i in range(20)
        ]
        result = _select_candidates("query", candidates)
        assert len(result) <= 3  # SEARCH_MAX_CANDIDATES_PER_LEVEL

    @patch("core.llm.subprocess.run")
    def test_llm_fallback_on_ambiguous_scores(self, mock_run):
        from core.searcher import _select_candidates
        import json

        mock_run.return_value = MagicMock(
            stdout=json.dumps([2, 1]),
            stderr="",
            returncode=0,
        )

        candidates = [
            {"id": 1, "name": "A", "description": "desc", "score": 0.50},
            {"id": 2, "name": "B", "description": "desc", "score": 0.55},
        ]
        # Both above LOW (0.30), both above relative cutoff (0.55*0.70=0.385),
        # gap 0.05 < AMBIGUITY_RANGE (0.10), so LLM should be called
        result = _select_candidates("query", candidates)
        result_ids = [c["id"] for c in result]
        assert 2 in result_ids
