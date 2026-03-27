"""Unit tests for core.embedder — all HTTP calls are mocked."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest

from core.embedder import embed_documents, embed_query, _call_jina_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, n_items: int = 1, n_tokens: int = 10):
    """Build a mock requests.Response with realistic Jina API payload."""
    resp = MagicMock()
    resp.status_code = status_code
    if status_code == 200:
        data = [
            {"embeddings": np.random.randn(n_tokens, 128).tolist()}
            for _ in range(n_items)
        ]
        resp.json.return_value = {"data": data}
    else:
        resp.text = f"Error {status_code}"
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbedDocuments:
    """Tests for embed_documents."""

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_successful_response_shapes(self, _mock_key, mock_post):
        """Returned arrays should be 2-D with shape (N_tokens, 128)."""
        mock_post.return_value = _make_response(n_items=3, n_tokens=12)

        results = embed_documents(["a", "b", "c"])

        assert len(results) == 3
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (12, 128)

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_batching_20_texts(self, _mock_key, mock_post):
        """20 texts should produce 2 API calls (batch of 16 + batch of 4)."""
        mock_post.side_effect = [
            _make_response(n_items=16, n_tokens=8),
            _make_response(n_items=4, n_tokens=8),
        ]

        results = embed_documents([f"text_{i}" for i in range(20)])

        assert mock_post.call_count == 2
        assert len(results) == 20

        # First call should have 16 inputs, second should have 4
        first_body = mock_post.call_args_list[0][1]["json"]
        second_body = mock_post.call_args_list[1][1]["json"]
        assert len(first_body["input"]) == 16
        assert len(second_body["input"]) == 4


class TestEmbedQuery:
    """Tests for embed_query."""

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_returns_correct_shape(self, _mock_key, mock_post):
        """embed_query should return a single 2-D array (Q_tokens, 128)."""
        mock_post.return_value = _make_response(n_items=1, n_tokens=7)

        result = embed_query("what is colbert?")

        assert isinstance(result, np.ndarray)
        assert result.shape == (7, 128)


class TestAPIErrorHandling:
    """Tests for error paths."""

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_400_raises_value_error(self, _mock_key, mock_post):
        """Non-retryable errors (e.g. 400) should raise ValueError."""
        mock_post.return_value = _make_response(status_code=400)

        with pytest.raises(ValueError, match="400"):
            embed_documents(["bad input"])

    @patch("core.embedder.time.sleep")
    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_retry_on_429_then_success(self, _mock_key, mock_post, mock_sleep):
        """429 should trigger a retry; success on second attempt."""
        mock_post.side_effect = [
            _make_response(status_code=429),
            _make_response(n_items=1, n_tokens=5),
        ]

        result = embed_query("retry me")

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1)  # first backoff = 1s
        assert result.shape == (5, 128)

    @patch("core.embedder.time.sleep")
    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_retry_exhaustion_raises_connection_error(self, _mock_key, mock_post, mock_sleep):
        """Three consecutive 500s should raise ConnectionError."""
        mock_post.side_effect = [
            _make_response(status_code=500),
            _make_response(status_code=500),
            _make_response(status_code=500),
        ]

        with pytest.raises(ConnectionError):
            embed_documents(["fail"])

        assert mock_post.call_count == 3
        # Two sleeps (after attempt 0 and 1; no sleep after final attempt)
        assert mock_sleep.call_count == 2


class TestRequestPayload:
    """Verify the exact headers and body sent to the Jina API."""

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_headers_and_body_for_document(self, _mock_key, mock_post):
        mock_post.return_value = _make_response(n_items=1)

        embed_documents(["hello world"])

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args

        # Headers
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"

        # Body
        body = kwargs["json"]
        assert body["model"] == "jina-colbert-v2"
        assert body["input"] == ["hello world"]
        assert body["input_type"] == "document"
        assert body["dimensions"] == 128

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_headers_and_body_for_query(self, _mock_key, mock_post):
        mock_post.return_value = _make_response(n_items=1)

        embed_query("search term")

        _, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["input"] == ["search term"]
        assert body["input_type"] == "query"
