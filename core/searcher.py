"""MaxSim search over ColBERT multi-vector embeddings."""

import numpy as np
from tokenizers import Tokenizer

from core.store import get_all_chunk_ids, load_vectors, get_chunk
from core.embedder import embed_query
from config import DEFAULT_TOP_K

# Load the XLM-RoBERTa tokenizer once (jina-colbert-v2's backbone).
# This is just a vocabulary lookup table — no GPU, ~2MB memory.
_tokenizer: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained("xlm-roberta-base")
    return _tokenizer


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product == cosine similarity."""
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)


def maxsim_score(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
    """Compute the MaxSim score between query and document token embeddings.

    For each query token vector, find the maximum cosine similarity against all
    document token vectors, then sum those maxima.

    Args:
        query_vectors: Array of shape (Q, 128) — one row per query token.
        doc_vectors:   Array of shape (D, 128) — one row per document token.

    Returns:
        Scalar MaxSim score (sum of per-query-token max cosine similarities).
    """
    q_norm = _normalize(query_vectors)
    d_norm = _normalize(doc_vectors)
    sim = q_norm @ d_norm.T
    return float(np.sum(np.max(sim, axis=1)))


def token_relevance(
    query_vectors: np.ndarray, doc_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-document-token relevance to the query.

    For each document token, returns its max cosine similarity to any query
    token and the index of the best-matching query token.

    Args:
        query_vectors: Array of shape (Q, 128).
        doc_vectors:   Array of shape (D, 128).

    Returns:
        Tuple of (scores, matched_query_indices) — both shape (D,).
    """
    q_norm = _normalize(query_vectors)
    d_norm = _normalize(doc_vectors)
    sim = q_norm @ d_norm.T  # (Q, D)
    return np.max(sim, axis=0), np.argmax(sim, axis=0)


def _query_token_to_word(query: str, query_vectors: np.ndarray) -> list[str]:
    """Map each query API token index to the query word it belongs to.

    ColBERT query layout: [Q_marker] [<s>] [real tokens...] [</s>] [padding...]
    padded to a fixed length (typically 32).  The marker is an extra token not
    produced by the local tokenizer, so API index ``i`` corresponds to local
    tokenizer index ``i - 1``.  Indices beyond the tokenizer length are padding.

    Returns a list of length equal to ``len(query_vectors)`` where each element
    is the whitespace-split query word that token came from.  Special / padding
    tokens map to ``""``.
    """
    tokenizer = _get_tokenizer()
    q_words = query.split()
    if not q_words:
        return [""] * len(query_vectors)

    encoding = tokenizer.encode(query)
    offsets = encoding.offsets  # includes <s> and </s>

    # Build char→word map for the query
    word_starts: list[int] = []
    pos = 0
    for w in q_words:
        idx = query.find(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    def char_to_word_idx(char_pos: int) -> int:
        best = 0
        for i, ws in enumerate(word_starts):
            if ws <= char_pos:
                best = i
            else:
                break
        return best

    # API index 0 = [Q] marker (not in local tokenizer)
    # API index 1 = local index 0 (<s>, offset (0,0) → skip)
    # API index 2 = local index 1 (first real subword token)
    # ...
    # API index len(offsets) = local index len(offsets)-1 (</s>)
    # API index > len(offsets) = padding → ""
    marker_offset = 1  # the [Q] prefix token

    mapping: list[str] = []
    for t_idx in range(len(query_vectors)):
        tok_idx = t_idx - marker_offset
        if tok_idx < 0 or tok_idx >= len(offsets):
            mapping.append("")
            continue
        start, end = offsets[tok_idx]
        if start == 0 and end == 0:
            mapping.append("")
            continue
        mapping.append(q_words[char_to_word_idx(start)])

    return mapping


def word_relevance(
    text: str,
    query: str,
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
) -> list[tuple[str, float, str]]:
    """Map token-level relevance scores to words using the XLM-RoBERTa tokenizer.

    Uses the tokenizer's character offsets to map subword token scores back to
    exact word spans in the original text. Multiple subword tokens belonging to
    the same word are max-pooled; the matched query word comes from the
    best-scoring token.

    Returns:
        List of (word, score, matched_query_word) tuples matching
        whitespace-split words. matched_query_word is "" for low-relevance words.
    """
    words = text.split()
    if not words:
        return []

    tok_scores, tok_query_indices = token_relevance(query_vectors, doc_vectors)
    query_token_words = _query_token_to_word(query, query_vectors)

    # Tokenize to get character-level offsets
    tokenizer = _get_tokenizer()
    encoding = tokenizer.encode(text)
    offsets = encoding.offsets  # list of (start_char, end_char) per token

    # Build a map from character position to word index
    # We find the start char of each whitespace-split word
    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = text.find(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    def char_to_word_idx(char_pos: int) -> int:
        """Find which word a character position belongs to."""
        best = 0
        for i, ws in enumerate(word_starts):
            if ws <= char_pos:
                best = i
            else:
                break
        return best

    # Map each token's score to its word, max-pooling
    word_scores = np.zeros(len(words))
    word_matched_query: list[str] = [""] * len(words)

    # The API returns vectors including <s> and </s> special tokens.
    # The tokenizer also produces them. We align by index directly.
    # Skip special tokens (offset == (0,0)) when mapping.
    num_api_tokens = len(tok_scores)
    num_tokenizer_tokens = len(offsets)

    # If API has one extra token (e.g. query prefix), offset by 1
    offset_shift = max(0, num_api_tokens - num_tokenizer_tokens)

    for t_idx in range(num_api_tokens):
        tok_idx = t_idx - offset_shift
        if tok_idx < 0 or tok_idx >= num_tokenizer_tokens:
            continue

        start, end = offsets[tok_idx]
        if start == 0 and end == 0:
            # Special token (<s>, </s>) — skip
            continue

        w_idx = char_to_word_idx(start)
        score = tok_scores[t_idx]
        if score > word_scores[w_idx]:
            word_scores[w_idx] = score
            qi = int(tok_query_indices[t_idx])
            word_matched_query[w_idx] = query_token_words[qi] if qi < len(query_token_words) else ""

    return list(zip(words, word_scores.tolist(), word_matched_query))


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    document_id: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """Run a MaxSim search and return the top-k matching chunks.

    Args:
        query:       Natural-language query string.
        top_k:       Number of results to return.
        document_id: If set, restrict search to chunks from this document.
        section:     If set, restrict search to chunks in this section.

    Returns:
        List of result dicts sorted by score descending.  Each dict contains:
        chunk_id, score, text, section, document_id, chunk_index, token_count.
    """
    query_vectors = embed_query(query)
    chunk_ids = get_all_chunk_ids(document_id=document_id, section=section)

    if not chunk_ids:
        return []

    scored: list[tuple[int, float]] = []
    for cid in chunk_ids:
        doc_vectors = load_vectors(cid)
        score = maxsim_score(query_vectors, doc_vectors)
        scored.append((cid, score))

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    results: list[dict] = []
    for cid, score in scored:
        chunk = get_chunk(cid)
        if chunk is None:
            continue
        doc_vectors = load_vectors(cid)
        highlights = word_relevance(chunk["text"], query, query_vectors, doc_vectors)
        results.append({
            "chunk_id": cid,
            "score": score,
            "text": chunk["text"],
            "section": chunk["section"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"],
            "highlights": highlights,
        })

    return results
