"""Jina ColBERT-v2 multi-vector embedding client."""

import time

import numpy as np
import requests

from config import JINA_API_URL, JINA_MODEL, JINA_DIMENSIONS, get_jina_api_key

BATCH_SIZE = 16
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]


def _call_jina_api(texts: list[str], input_type: str) -> list[np.ndarray]:
    """Make an HTTP POST to the Jina multi-vector embedding API.

    Args:
        texts: List of text strings to embed.
        input_type: Either "document" or "query".

    Returns:
        List of 2-D numpy arrays, each with shape (N_tokens, 128).

    Raises:
        ValueError: If the API returns a non-200, non-retryable status.
        ConnectionError: If retries are exhausted on 429 / 5xx errors.
    """
    api_key = get_jina_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": JINA_MODEL,
        "input": texts,
        "input_type": input_type,
        "dimensions": JINA_DIMENSIONS,
    }

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response = requests.post(JINA_API_URL, headers=headers, json=body)

        if response.status_code == 200:
            data = response.json()["data"]
            return [
                np.array(item["embeddings"], dtype=np.float32) for item in data
            ]

        # Retryable status codes: 429 (rate-limit) and 5xx (server error)
        if response.status_code == 429 or response.status_code >= 500:
            last_error = ConnectionError(
                f"Jina API returned {response.status_code}: {response.text}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])
            continue

        # Non-retryable error
        raise ValueError(
            f"Jina API error {response.status_code}: {response.text}"
        )

    raise last_error  # type: ignore[misc]


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    """Embed document texts via Jina ColBERT-v2.

    Requests are batched in groups of 16 to respect API limits.

    Args:
        texts: Document strings to embed.

    Returns:
        List of 2-D numpy arrays, each shape (N_tokens, 128).
    """
    results: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        results.extend(_call_jina_api(batch, input_type="document"))
    return results


def embed_query(query: str) -> np.ndarray:
    """Embed a single query via Jina ColBERT-v2.

    Args:
        query: The query string.

    Returns:
        2-D numpy array of shape (Q_tokens, 128).
    """
    arrays = _call_jina_api([query], input_type="query")
    return arrays[0]
