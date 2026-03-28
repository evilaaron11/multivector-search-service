"""Jina embedding client — multi-vector (ColBERT) and single-vector modes."""

import time

import numpy as np
import requests

from config import (
    JINA_API_URL,
    JINA_MODEL,
    JINA_DIMENSIONS,
    JINA_SINGLE_VECTOR_URL,
    JINA_SINGLE_VECTOR_MODEL,
    JINA_SINGLE_VECTOR_DIMENSIONS,
    get_jina_api_key,
)

BATCH_SIZE = 16
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _api_call_with_retry(url: str, body: dict) -> dict:
    """POST to a Jina API endpoint with retry on 429/5xx.

    Returns the parsed JSON response body on success.
    """
    api_key = get_jina_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429 or response.status_code >= 500:
            last_error = ConnectionError(
                f"Jina API returned {response.status_code}: {response.text}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])
            continue

        raise ValueError(
            f"Jina API error {response.status_code}: {response.text}"
        )

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Multi-vector (ColBERT) embeddings
# ---------------------------------------------------------------------------

def _call_jina_multivector(texts: list[str], input_type: str) -> list[np.ndarray]:
    """Call Jina ColBERT-v2 multi-vector API.

    Returns list of 2-D arrays, each shape (N_tokens, 128).
    """
    body = {
        "model": JINA_MODEL,
        "input": texts,
        "input_type": input_type,
        "dimensions": JINA_DIMENSIONS,
    }
    data = _api_call_with_retry(JINA_API_URL, body)["data"]
    return [np.array(item["embeddings"], dtype=np.float32) for item in data]


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    """Embed document texts via Jina ColBERT-v2 (multi-vector).

    Batched in groups of 16. Returns list of 2-D arrays, each shape (N_tokens, 128).
    """
    results: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        results.extend(_call_jina_multivector(batch, input_type="document"))
    return results


def embed_query(query: str) -> np.ndarray:
    """Embed a single query via Jina ColBERT-v2 (multi-vector).

    Returns 2-D array of shape (Q_tokens, 128).
    """
    arrays = _call_jina_multivector([query], input_type="query")
    return arrays[0]


# ---------------------------------------------------------------------------
# Single-vector embeddings
# ---------------------------------------------------------------------------

def _call_jina_single_vector(texts: list[str], input_type: str) -> list[np.ndarray]:
    """Call Jina embeddings v3 single-vector API.

    Returns list of 1-D arrays, each shape (dimensions,).
    """
    body = {
        "model": JINA_SINGLE_VECTOR_MODEL,
        "input": texts,
        "input_type": input_type,
        "dimensions": JINA_SINGLE_VECTOR_DIMENSIONS,
    }
    data = _api_call_with_retry(JINA_SINGLE_VECTOR_URL, body)["data"]
    return [np.array(item["embedding"], dtype=np.float32) for item in data]


def embed_single_vector(texts: list[str], input_type: str = "document") -> list[np.ndarray]:
    """Embed texts as single vectors via Jina v3.

    Batched in groups of 16. Returns list of 1-D arrays, each shape (dimensions,).
    """
    results: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        results.extend(_call_jina_single_vector(batch, input_type=input_type))
    return results


def embed_single_query(query: str) -> np.ndarray:
    """Embed a single query as a single vector.

    Returns 1-D array of shape (dimensions,).
    """
    arrays = _call_jina_single_vector([query], input_type="query")
    return arrays[0]
