"""Search: MaxSim for multi-vector, cosine for single-vector, graph traversal."""

import numpy as np
from tokenizers import Tokenizer

from core.store import get_node, get_leaf_nodes, get_roots, get_children, load_vectors
from core.embedder import embed_query, embed_single_query
from core.graph import get_graph_description
from config import (
    DEFAULT_TOP_K,
    SEARCH_SCORE_THRESHOLD_HIGH,
    SEARCH_SCORE_THRESHOLD_LOW,
    SEARCH_MAX_CANDIDATES_PER_LEVEL,
    SEARCH_LLM_AMBIGUITY_RANGE,
)

# Load the XLM-RoBERTa tokenizer once (jina-colbert-v2's backbone).
_tokenizer: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained("xlm-roberta-base")
    return _tokenizer


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product == cosine similarity."""
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def maxsim_score(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
    """MaxSim score between query and document token embeddings.

    For each query token, find max cosine similarity against all document tokens,
    then sum those maxima.
    """
    q_norm = _normalize(query_vectors)
    d_norm = _normalize(doc_vectors)
    sim = q_norm @ d_norm.T
    return float(np.sum(np.max(sim, axis=1)))


def cosine_score(query_vector: np.ndarray, node_vector: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Used for single-vector nodes."""
    q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    d = node_vector / (np.linalg.norm(node_vector) + 1e-10)
    return float(q @ d)


def score_node(
    query_single: np.ndarray,
    query_multi: np.ndarray,
    node_id: int,
    embedding_type: str,
) -> float:
    """Score a node against query vectors, dispatching based on embedding type."""
    vectors = load_vectors(node_id)
    if embedding_type == "single":
        return cosine_score(query_single, vectors)
    else:  # "multi"
        return maxsim_score(query_multi, vectors)


# ---------------------------------------------------------------------------
# Token-level relevance (unchanged from original)
# ---------------------------------------------------------------------------

def token_relevance(
    query_vectors: np.ndarray, doc_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document-token relevance: max cosine sim to any query token."""
    q_norm = _normalize(query_vectors)
    d_norm = _normalize(doc_vectors)
    sim = q_norm @ d_norm.T  # (Q, D)
    return np.max(sim, axis=0), np.argmax(sim, axis=0)


def _query_token_to_word(query: str, query_vectors: np.ndarray) -> list[str]:
    """Map each query API token index to the query word it belongs to."""
    tokenizer = _get_tokenizer()
    q_words = query.split()
    if not q_words:
        return [""] * len(query_vectors)

    encoding = tokenizer.encode(query)
    offsets = encoding.offsets

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

    marker_offset = 1
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
    """Map token-level relevance scores to words using XLM-RoBERTa tokenizer."""
    words = text.split()
    if not words:
        return []

    tok_scores, tok_query_indices = token_relevance(query_vectors, doc_vectors)
    query_token_words = _query_token_to_word(query, query_vectors)

    tokenizer = _get_tokenizer()
    encoding = tokenizer.encode(text)
    offsets = encoding.offsets

    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = text.find(w, pos)
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

    word_scores = np.zeros(len(words))
    word_matched_query: list[str] = [""] * len(words)

    num_api_tokens = len(tok_scores)
    num_tokenizer_tokens = len(offsets)
    offset_shift = max(0, num_api_tokens - num_tokenizer_tokens)

    for t_idx in range(num_api_tokens):
        tok_idx = t_idx - offset_shift
        if tok_idx < 0 or tok_idx >= num_tokenizer_tokens:
            continue
        start, end = offsets[tok_idx]
        if start == 0 and end == 0:
            continue
        w_idx = char_to_word_idx(start)
        score = tok_scores[t_idx]
        if score > word_scores[w_idx]:
            word_scores[w_idx] = score
            qi = int(tok_query_indices[t_idx])
            word_matched_query[w_idx] = query_token_words[qi] if qi < len(query_token_words) else ""

    return list(zip(words, word_scores.tolist(), word_matched_query))


# ---------------------------------------------------------------------------
# Flat search (backward-compatible, searches all leaf nodes)
# ---------------------------------------------------------------------------

def flat_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    node_id: int | None = None,
) -> list[dict]:
    """Run a MaxSim search over all leaf nodes (or leaves under a specific node).

    This is the original search behavior, preserved for backward compatibility
    and as a fallback when graph navigation isn't possible.
    """
    query_vectors = embed_query(query)
    leaves = get_leaf_nodes(parent_id=node_id)

    if not leaves:
        return []

    scored: list[tuple[dict, float]] = []
    for leaf in leaves:
        if leaf.get("embedding_type") != "multi":
            continue
        doc_vectors = load_vectors(leaf["id"])
        score = maxsim_score(query_vectors, doc_vectors)
        scored.append((leaf, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    results: list[dict] = []
    for leaf, score in scored:
        doc_vectors = load_vectors(leaf["id"])
        highlights = word_relevance(leaf["text"], query, query_vectors, doc_vectors)
        results.append({
            "node_id": leaf["id"],
            "score": score,
            "text": leaf["text"],
            "name": leaf["name"],
            "node_type": leaf["node_type"],
            "token_count": leaf["token_count"],
            "highlights": highlights,
        })

    return results


# ---------------------------------------------------------------------------
# Graph search (level-by-level DAG navigation)
# ---------------------------------------------------------------------------

def _select_candidates(
    query: str,
    candidates: list[dict],
) -> list[dict]:
    """Apply threshold routing to select which candidates to descend into.

    Uses score thresholds with LLM fallback for ambiguous cases.
    """
    if not candidates:
        return []

    scores = [c["score"] for c in candidates]
    max_score = max(scores)
    min_score = min(scores)

    # Filter to candidates above the low threshold
    viable = [c for c in candidates if c["score"] >= SEARCH_SCORE_THRESHOLD_LOW]
    if not viable:
        # Nothing above threshold — take the single best anyway
        return [candidates[0]]

    # If scores are tightly clustered (ambiguous), use LLM routing
    if max_score - min_score <= SEARCH_LLM_AMBIGUITY_RANGE and len(viable) > 1:
        try:
            from core.llm import llm_route_search
            selected_ids = llm_route_search(query, viable)
            selected = [c for c in viable if c["id"] in selected_ids]
            if selected:
                return selected[:SEARCH_MAX_CANDIDATES_PER_LEVEL]
        except Exception:
            pass  # Fall through to threshold-based selection

    # Take candidates above high threshold, or top N if none are that high
    high = [c for c in viable if c["score"] >= SEARCH_SCORE_THRESHOLD_HIGH]
    if high:
        return high[:SEARCH_MAX_CANDIDATES_PER_LEVEL]

    return viable[:SEARCH_MAX_CANDIDATES_PER_LEVEL]


def graph_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Navigate the DAG level by level to find relevant leaf nodes.

    1. Embed query as both single-vector and multi-vector
    2. Start at roots, score with appropriate method
    3. Threshold routing to pick candidates at each level
    4. Descend until reaching leaves
    5. Score leaves with MaxSim, return top-k with highlights
    """
    roots = get_roots()
    if not roots:
        return []

    # Check if any roots have embeddings — if not, fall back to flat search
    roots_with_embeddings = [r for r in roots if r.get("embedding_type")]
    if not roots_with_embeddings:
        return flat_search(query, top_k=top_k)

    # Embed query in both modes
    query_single = embed_single_query(query)
    query_multi = embed_query(query)

    # Start at roots — score them
    current_level = roots_with_embeddings
    leaf_candidates: list[tuple[dict, float]] = []

    visited = set()
    max_depth = 10  # safety limit

    for _ in range(max_depth):
        if not current_level:
            break

        # Score each node at this level
        scored_nodes: list[dict] = []
        for node in current_level:
            if node["id"] in visited:
                continue
            visited.add(node["id"])

            if not node.get("embedding_type"):
                # No embedding — descend into children blindly
                children = get_children(node["id"])
                for child in children:
                    if child.get("embedding_type"):
                        scored_nodes.append({**child, "score": 0.5})
                    else:
                        # Leaf without embedding — can't score, skip
                        pass
                continue

            score = score_node(query_single, query_multi, node["id"], node["embedding_type"])
            scored_nodes.append({**node, "score": score})

        if not scored_nodes:
            break

        # Separate leaves from branch nodes
        next_level: list[dict] = []
        for node in scored_nodes:
            children = get_children(node["id"])
            if not children:
                # Leaf node — collect for final scoring
                leaf_candidates.append((node, node["score"]))
            else:
                next_level.append(node)

        # Select which branch nodes to descend into
        if next_level:
            next_level.sort(key=lambda x: x["score"], reverse=True)
            selected = _select_candidates(query, next_level)

            # Get children of selected nodes for next iteration
            current_level = []
            for node in selected:
                current_level.extend(get_children(node["id"]))
        else:
            current_level = []

    # Re-score leaf nodes with full MaxSim if they were scored with single-vector
    final_scored: list[tuple[dict, float]] = []
    for leaf, _ in leaf_candidates:
        if leaf.get("embedding_type") == "multi":
            doc_vectors = load_vectors(leaf["id"])
            score = maxsim_score(query_multi, doc_vectors)
            final_scored.append((leaf, score))
        elif leaf.get("embedding_type") == "single":
            vectors = load_vectors(leaf["id"])
            score = cosine_score(query_single, vectors)
            final_scored.append((leaf, score))

    final_scored.sort(key=lambda x: x[1], reverse=True)
    final_scored = final_scored[:top_k]

    # Build results with highlights
    results: list[dict] = []
    for leaf, score in final_scored:
        result = {
            "node_id": leaf["id"],
            "score": score,
            "text": leaf.get("text", ""),
            "name": leaf["name"],
            "node_type": leaf["node_type"],
            "token_count": leaf.get("token_count", 0),
        }

        # Only do word highlighting for multi-vector leaves
        if leaf.get("embedding_type") == "multi" and leaf.get("text"):
            doc_vectors = load_vectors(leaf["id"])
            result["highlights"] = word_relevance(leaf["text"], query, query_multi, doc_vectors)
        else:
            result["highlights"] = []

        results.append(result)

    return results
