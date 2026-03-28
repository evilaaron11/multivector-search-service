"""Unit tests for core.embedder — multi-vector and single-vector, all HTTP mocked."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.embedder import (
    embed_documents,
    embed_query,
    embed_single_vector,
    embed_single_query,
    _api_call_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multivector_response(status_code=200, n_items=1, n_tokens=10):
    """Mock Jina multi-vector API response."""
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


def _make_single_vector_response(status_code=200, n_items=1, dimensions=1024):
    """Mock Jina single-vector API response."""
    resp = MagicMock()
    resp.status_code = status_code
    if status_code == 200:
        data = [
            {"embedding": np.random.randn(dimensions).tolist()}
            for _ in range(n_items)
        ]
        resp.json.return_value = {"data": data}
    else:
        resp.text = f"Error {status_code}"
    return resp


# ---------------------------------------------------------------------------
# Multi-vector (ColBERT) tests
# ---------------------------------------------------------------------------

class TestEmbedDocuments:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_successful_response_shapes(self, _key, mock_post):
        mock_post.return_value = _make_multivector_response(n_items=3, n_tokens=12)
        results = embed_documents(["a", "b", "c"])
        assert len(results) == 3
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (12, 128)

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_batching_20_texts(self, _key, mock_post):
        mock_post.side_effect = [
            _make_multivector_response(n_items=16, n_tokens=8),
            _make_multivector_response(n_items=4, n_tokens=8),
        ]
        results = embed_documents([f"text_{i}" for i in range(20)])
        assert mock_post.call_count == 2
        assert len(results) == 20

        first_body = mock_post.call_args_list[0][1]["json"]
        second_body = mock_post.call_args_list[1][1]["json"]
        assert len(first_body["input"]) == 16
        assert len(second_body["input"]) == 4


class TestEmbedQuery:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_returns_correct_shape(self, _key, mock_post):
        mock_post.return_value = _make_multivector_response(n_items=1, n_tokens=7)
        result = embed_query("what is colbert?")
        assert isinstance(result, np.ndarray)
        assert result.shape == (7, 128)


# ---------------------------------------------------------------------------
# Single-vector tests
# ---------------------------------------------------------------------------

class TestEmbedSingleVector:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_successful_response_shapes(self, _key, mock_post):
        mock_post.return_value = _make_single_vector_response(n_items=2, dimensions=1024)
        results = embed_single_vector(["hello", "world"])
        assert len(results) == 2
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 1
            assert arr.shape == (1024,)

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_batching(self, _key, mock_post):
        mock_post.side_effect = [
            _make_single_vector_response(n_items=16, dimensions=1024),
            _make_single_vector_response(n_items=2, dimensions=1024),
        ]
        results = embed_single_vector([f"t{i}" for i in range(18)])
        assert mock_post.call_count == 2
        assert len(results) == 18

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_payload_uses_correct_model_and_url(self, _key, mock_post):
        mock_post.return_value = _make_single_vector_response(n_items=1)
        embed_single_vector(["test"], input_type="document")

        _, kwargs = mock_post.call_args
        assert "embeddings" in kwargs.get("url", mock_post.call_args[0][0] if mock_post.call_args[0] else "")
        body = kwargs["json"]
        assert body["model"] == "jina-embeddings-v3"
        assert body["input_type"] == "document"
        assert body["dimensions"] == 1024


class TestEmbedSingleQuery:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_returns_1d_array(self, _key, mock_post):
        mock_post.return_value = _make_single_vector_response(n_items=1, dimensions=1024)
        result = embed_single_query("search term")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (1024,)

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_payload_input_type_is_query(self, _key, mock_post):
        mock_post.return_value = _make_single_vector_response(n_items=1)
        embed_single_query("find me")
        body = mock_post.call_args[1]["json"]
        assert body["input_type"] == "query"


# ---------------------------------------------------------------------------
# Shared retry logic
# ---------------------------------------------------------------------------

class TestAPIErrorHandling:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_400_raises_value_error(self, _key, mock_post):
        mock_post.return_value = _make_multivector_response(status_code=400)
        with pytest.raises(ValueError, match="400"):
            embed_documents(["bad input"])

    @patch("core.embedder.time.sleep")
    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_retry_on_429_then_success(self, _key, mock_post, mock_sleep):
        mock_post.side_effect = [
            _make_multivector_response(status_code=429),
            _make_multivector_response(n_items=1, n_tokens=5),
        ]
        result = embed_query("retry me")
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1)
        assert result.shape == (5, 128)

    @patch("core.embedder.time.sleep")
    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_retry_exhaustion_raises_connection_error(self, _key, mock_post, mock_sleep):
        mock_post.side_effect = [
            _make_multivector_response(status_code=500),
            _make_multivector_response(status_code=500),
            _make_multivector_response(status_code=500),
        ]
        with pytest.raises(ConnectionError):
            embed_documents(["fail"])
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("core.embedder.time.sleep")
    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_single_vector_retry(self, _key, mock_post, mock_sleep):
        """Single-vector endpoint also retries on 429."""
        mock_post.side_effect = [
            _make_single_vector_response(status_code=429),
            _make_single_vector_response(n_items=1, dimensions=1024),
        ]
        result = embed_single_query("retry me")
        assert mock_post.call_count == 2
        assert result.shape == (1024,)


class TestRequestPayload:

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_multivector_headers_and_body(self, _key, mock_post):
        mock_post.return_value = _make_multivector_response(n_items=1)
        embed_documents(["hello world"])

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        body = kwargs["json"]
        assert body["model"] == "jina-colbert-v2"
        assert body["input"] == ["hello world"]
        assert body["input_type"] == "document"
        assert body["dimensions"] == 128

    @patch("core.embedder.requests.post")
    @patch("core.embedder.get_jina_api_key", return_value="test-key")
    def test_multivector_query_body(self, _key, mock_post):
        mock_post.return_value = _make_multivector_response(n_items=1)
        embed_query("search term")
        body = mock_post.call_args[1]["json"]
        assert body["input_type"] == "query"
