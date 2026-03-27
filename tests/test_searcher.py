"""Unit tests for core.searcher — MaxSim search over ColBERT multi-vectors."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.searcher import maxsim_score, search


# ---------------------------------------------------------------------------
# maxsim_score tests
# ---------------------------------------------------------------------------

class TestMaxSimScore:
    """Tests for the maxsim_score function."""

    def test_identical_vectors_give_high_score(self):
        """Identical normalised vectors → each max cosine sim is 1.0, so score = Q."""
        vecs = np.eye(3, 128, dtype=np.float32)  # 3 orthonormal rows
        score = maxsim_score(vecs, vecs)
        assert score == pytest.approx(3.0, abs=1e-5)

    def test_orthogonal_vectors_give_zero(self):
        """Query and doc vectors that are mutually orthogonal → score ≈ 0."""
        query = np.zeros((2, 128), dtype=np.float32)
        query[0, 0] = 1.0
        query[1, 1] = 1.0

        doc = np.zeros((2, 128), dtype=np.float32)
        doc[0, 2] = 1.0
        doc[1, 3] = 1.0

        score = maxsim_score(query, doc)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_non_square_shapes(self):
        """query (3, 128) vs doc (10, 128) — different token counts."""
        rng = np.random.default_rng(42)
        query = rng.standard_normal((3, 128)).astype(np.float32)
        doc = rng.standard_normal((10, 128)).astype(np.float32)

        # Manually compute expected score
        q_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
        d_norm = doc / np.linalg.norm(doc, axis=1, keepdims=True)
        sim = q_norm @ d_norm.T
        expected = float(np.sum(np.max(sim, axis=1)))

        score = maxsim_score(query, doc)
        assert score == pytest.approx(expected, abs=1e-5)

    def test_normalization_of_non_unit_vectors(self):
        """Vectors that are NOT unit-length should still produce correct cosine sims."""
        # Two parallel vectors with different magnitudes → cosine sim = 1.0
        query = np.zeros((1, 128), dtype=np.float32)
        query[0, 0] = 7.0  # not unit length

        doc = np.zeros((1, 128), dtype=np.float32)
        doc[0, 0] = 0.3  # also not unit length, but same direction

        score = maxsim_score(query, doc)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_normalization_scaled_vectors_match(self):
        """Scaling vectors arbitrarily should not change the score."""
        rng = np.random.default_rng(99)
        query = rng.standard_normal((4, 128)).astype(np.float32)
        doc = rng.standard_normal((6, 128)).astype(np.float32)

        score_original = maxsim_score(query, doc)
        score_scaled = maxsim_score(query * 100.0, doc * 0.01)
        assert score_original == pytest.approx(score_scaled, abs=1e-4)


# ---------------------------------------------------------------------------
# search() tests
# ---------------------------------------------------------------------------

def _make_chunk_meta(chunk_id, document_id=1, section="intro", chunk_index=0, token_count=50):
    return {
        "id": chunk_id,
        "document_id": document_id,
        "text": f"Text for chunk {chunk_id}",
        "section": section,
        "chunk_index": chunk_index,
        "token_count": token_count,
    }


class TestSearch:
    """Tests for the search() function."""

    @patch("core.searcher.get_chunk")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_all_chunk_ids")
    @patch("core.searcher.embed_query")
    def test_results_sorted_by_score_descending(
        self, mock_embed, mock_ids, mock_load, mock_chunk
    ):
        rng = np.random.default_rng(1)
        query_vecs = rng.standard_normal((4, 128)).astype(np.float32)
        mock_embed.return_value = query_vecs

        # Three chunks with vectors designed to produce different scores
        mock_ids.return_value = [10, 20, 30]

        # Chunk 10: random vectors (medium score)
        vecs_10 = rng.standard_normal((5, 128)).astype(np.float32)
        # Chunk 20: close to query vectors (high score)
        vecs_20 = query_vecs + rng.standard_normal((4, 128)).astype(np.float32) * 0.01
        # Chunk 30: orthogonal-ish (low score)
        vecs_30 = np.zeros((3, 128), dtype=np.float32)
        vecs_30[:, 64:67] = rng.standard_normal((3, 3)).astype(np.float32)

        def load_side_effect(cid):
            return {10: vecs_10, 20: vecs_20, 30: vecs_30}[cid]

        mock_load.side_effect = load_side_effect
        mock_chunk.side_effect = lambda cid: _make_chunk_meta(cid)

        results = search("test query", top_k=10)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["chunk_id"] == 20  # highest score

    @patch("core.searcher.get_chunk")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_all_chunk_ids")
    @patch("core.searcher.embed_query")
    def test_top_k_limit(self, mock_embed, mock_ids, mock_load, mock_chunk):
        rng = np.random.default_rng(2)
        mock_embed.return_value = rng.standard_normal((3, 128)).astype(np.float32)
        mock_ids.return_value = [1, 2, 3, 4, 5]
        mock_load.return_value = rng.standard_normal((4, 128)).astype(np.float32)
        mock_chunk.side_effect = lambda cid: _make_chunk_meta(cid)

        results = search("test", top_k=2)
        assert len(results) == 2

    @patch("core.searcher.get_chunk")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_all_chunk_ids")
    @patch("core.searcher.embed_query")
    def test_filters_passed_through(self, mock_embed, mock_ids, mock_load, mock_chunk):
        rng = np.random.default_rng(3)
        mock_embed.return_value = rng.standard_normal((2, 128)).astype(np.float32)
        mock_ids.return_value = [42]
        mock_load.return_value = rng.standard_normal((5, 128)).astype(np.float32)
        mock_chunk.side_effect = lambda cid: _make_chunk_meta(cid, document_id=7, section="methods")

        search("query", document_id=7, section="methods")

        mock_ids.assert_called_once_with(document_id=7, section="methods")

    @patch("core.searcher.get_all_chunk_ids")
    @patch("core.searcher.embed_query")
    def test_empty_when_no_chunks(self, mock_embed, mock_ids):
        mock_embed.return_value = np.random.default_rng(0).standard_normal((2, 128)).astype(np.float32)
        mock_ids.return_value = []

        results = search("anything")
        assert results == []

    @patch("core.searcher.get_chunk")
    @patch("core.searcher.load_vectors")
    @patch("core.searcher.get_all_chunk_ids")
    @patch("core.searcher.embed_query")
    def test_result_dict_keys(self, mock_embed, mock_ids, mock_load, mock_chunk):
        """Each result dict must contain the required keys."""
        rng = np.random.default_rng(4)
        mock_embed.return_value = rng.standard_normal((2, 128)).astype(np.float32)
        mock_ids.return_value = [1]
        mock_load.return_value = rng.standard_normal((3, 128)).astype(np.float32)
        mock_chunk.return_value = _make_chunk_meta(1)

        results = search("hello")
        assert len(results) == 1
        expected_keys = {"chunk_id", "score", "text", "section", "document_id", "chunk_index", "token_count", "highlights"}
        assert set(results[0].keys()) == expected_keys
