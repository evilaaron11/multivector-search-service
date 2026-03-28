"""Orchestration layer for the DAG-based embedding pipeline.

Ties together chunking, storage, embedding, LLM routing, and search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from core import store, chunker, embedder, searcher, edgar_fetcher, web_fetcher
from core.llm import (
    generate_node_summary,
    generate_leaf_description,
    propose_placement,
    confirm_duplicate,
)
from core.graph import get_graph_description
from config import DEDUP_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    node_id: int
    name: str
    num_chunks: int
    sections: list[str]
    parent_node_id: int | None = None
    is_new_branch: bool = False


@dataclass
class SearchResult:
    node_id: int
    score: float
    text: str
    name: str
    node_type: str
    highlights: list[tuple[str, float, str]] = field(default_factory=list)


@dataclass
class NodeInfo:
    node_id: int
    name: str
    node_type: str
    description: str
    child_count: int
    embedding_type: str | None = None


# Type alias for the interactive confirmation callback.
# It receives a question string and list of option dicts, returns the chosen index.
ConfirmCallback = Callable[[str, list[dict]], int]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _find_best_parent(
    summary: str,
    summary_vector: np.ndarray,
    confirm_callback: ConfirmCallback | None = None,
) -> int | None:
    """Find the best parent node for new content using embedding similarity + LLM.

    Returns the parent node_id, or None if content should become a new root.
    """
    store.init_db()

    # Get all non-leaf nodes that have embeddings (potential parents)
    all_nodes = store.get_all_nodes()
    parent_candidates = []
    for node in all_nodes:
        if node.get("embedding_type") == "single":
            vec = store.load_vectors(node["id"])
            score = searcher.cosine_score(summary_vector, vec)
            parent_candidates.append({
                "id": node["id"],
                "name": node["name"],
                "description": node.get("description", ""),
                "score": score,
            })

    if not parent_candidates:
        return None

    parent_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Get graph description for LLM context
    graph_desc = get_graph_description()

    # Ask LLM where to place this
    placement = propose_placement(summary, graph_desc, parent_candidates[:10])

    if placement["action"] == "attach_to_existing" and placement.get("target_node_id"):
        target_id = placement["target_node_id"]

        # If interactive and confidence is low, confirm with user
        if confirm_callback and placement.get("confidence", 1.0) < 0.7:
            options = [
                {"label": f"Attach to \"{c['name']}\" (score: {c['score']:.2f})", "node_id": c["id"]}
                for c in parent_candidates[:5]
            ] + [{"label": "Create new top-level category", "node_id": None}]

            choice = confirm_callback(
                f"Where should this content be placed? (LLM suggests: \"{parent_candidates[0]['name']}\")",
                options,
            )
            if choice < len(options):
                target_id = options[choice]["node_id"]

        return target_id

    elif placement["action"] == "create_new_branch":
        # Create a new category/branch node
        branch_name = placement.get("new_branch_name", "Uncategorized")
        branch_desc = placement.get("new_branch_description", "")
        branch_id = store.insert_node(
            node_type="category",
            name=branch_name,
            description=branch_desc,
        )
        # Embed the branch description
        if branch_desc:
            vec = embedder.embed_single_vector([branch_desc])[0]
            store.save_vectors(branch_id, vec)
            store.update_node(branch_id, embedding_type="single")
        return branch_id

    return None


def _check_duplicates(
    summary: str,
    summary_vector: np.ndarray,
) -> bool:
    """Check if this content is a duplicate of something already ingested.

    Returns True if duplicate found (should skip ingestion).
    """
    doc_nodes = store.get_all_nodes(node_type="document")
    for node in doc_nodes:
        if node.get("embedding_type") != "single":
            continue
        vec = store.load_vectors(node["id"])
        score = searcher.cosine_score(summary_vector, vec)
        if score >= DEDUP_SIMILARITY_THRESHOLD:
            result = confirm_duplicate(summary, node.get("description", node["name"]), score)
            if result.get("is_duplicate"):
                return True
    return False


def _force_delete_by_name(name: str) -> None:
    """Delete an existing document node (and descendants) matching the given name."""
    for node in store.get_all_nodes(node_type="document"):
        if node["name"] == name:
            store.delete_node(node["id"])
            return


def _force_delete_by_url(url: str) -> None:
    """Delete an existing document node (and descendants) matching a source URL."""
    existing = store.find_node_by_source_url(url)
    if existing:
        store.delete_node(existing["id"])


def ingest_document(
    file_path: str,
    confirm_callback: ConfirmCallback | None = None,
    force: bool = False,
) -> IngestResult:
    """Parse, chunk, embed, and store a document in the DAG.

    Args:
        file_path: Path to the markdown file.
        confirm_callback: If provided, called for ambiguous decisions (interactive mode).
            If None, all decisions are made autonomously.
        force: If True, delete existing document with same name before re-ingesting.

    Returns an IngestResult.
    """
    store.init_db()

    # Parse and chunk
    chunks = chunker.process_markdown(file_path)
    filename = Path(file_path).name

    # Force: delete existing document with same name
    if force:
        _force_delete_by_name(filename)

    # Collect chunk texts
    chunk_texts = [c["text"] for c in chunks]
    full_text = "\n\n".join(chunk_texts[:5])  # first 5 chunks for summary

    # Generate a summary of the document via LLM
    doc_summary = generate_leaf_description(full_text)

    # Embed the summary as a single vector
    summary_vector = embedder.embed_single_vector([doc_summary])[0]

    # Check for duplicates (skip if force)
    if not force and _check_duplicates(doc_summary, summary_vector):
        # Return early — duplicate found
        return IngestResult(
            node_id=-1,
            name=filename,
            num_chunks=0,
            sections=[],
            parent_node_id=None,
            is_new_branch=False,
        )

    # Find placement in the graph
    parent_id = _find_best_parent(doc_summary, summary_vector, confirm_callback)

    # Create document node
    doc_node_id = store.insert_node(
        node_type="document",
        name=filename,
        description=doc_summary,
        embedding_type="single",
    )
    store.save_vectors(doc_node_id, summary_vector)

    # Attach to parent if found
    is_new_branch = False
    if parent_id is not None:
        store.insert_edge(parent_id, doc_node_id)
        # Check if the parent was just created (new branch)
        parent = store.get_node(parent_id)
        if parent and parent["node_type"] == "category":
            is_new_branch = True

    # Embed all chunk texts (multi-vector ColBERT)
    embeddings = embedder.embed_documents(chunk_texts)

    # Create chunk nodes and edges
    seen_sections: set[str] = set()
    sections: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_node_id = store.insert_node(
            node_type="chunk",
            name=f"{chunk['section']} (chunk {chunk['chunk_index']})",
            text=chunk["text"],
            embedding_type="multi",
            token_count=chunk["token_count"],
        )
        store.save_vectors(chunk_node_id, embeddings[i])
        store.insert_edge(doc_node_id, chunk_node_id, sort_order=chunk["chunk_index"])

        sec = chunk["section"]
        if sec not in seen_sections:
            seen_sections.add(sec)
            sections.append(sec)

    # Update ancestor summaries
    update_ancestor_summaries(doc_node_id)

    return IngestResult(
        node_id=doc_node_id,
        name=filename,
        num_chunks=len(chunks),
        sections=sections,
        parent_node_id=parent_id,
        is_new_branch=is_new_branch,
    )


def ingest_ticker(
    ticker: str,
    confirm_callback: ConfirmCallback | None = None,
    force: bool = False,
) -> IngestResult:
    """Fetch SEC filing, chunk, embed, and store in the DAG."""
    store.init_db()

    chunks, metadata = edgar_fetcher.process_ticker(ticker)
    filename = f"{metadata['ticker']}_10K_{metadata['filing_date']}"

    if force:
        _force_delete_by_name(filename)

    chunk_texts = [c["text"] for c in chunks]
    full_text = "\n\n".join(chunk_texts[:5])

    doc_summary = generate_leaf_description(full_text)
    summary_vector = embedder.embed_single_vector([doc_summary])[0]

    if not force and _check_duplicates(doc_summary, summary_vector):
        return IngestResult(
            node_id=-1, name=filename, num_chunks=0, sections=[],
        )

    parent_id = _find_best_parent(doc_summary, summary_vector, confirm_callback)

    doc_node_id = store.insert_node(
        node_type="document",
        name=filename,
        description=doc_summary,
        embedding_type="single",
    )
    store.save_vectors(doc_node_id, summary_vector)

    is_new_branch = False
    if parent_id is not None:
        store.insert_edge(parent_id, doc_node_id)
        parent = store.get_node(parent_id)
        if parent and parent["node_type"] == "category":
            is_new_branch = True

    embeddings = embedder.embed_documents(chunk_texts)

    seen_sections: set[str] = set()
    sections: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_node_id = store.insert_node(
            node_type="chunk",
            name=f"{chunk['section']} (chunk {chunk['chunk_index']})",
            text=chunk["text"],
            embedding_type="multi",
            token_count=chunk["token_count"],
        )
        store.save_vectors(chunk_node_id, embeddings[i])
        store.insert_edge(doc_node_id, chunk_node_id, sort_order=chunk["chunk_index"])

        sec = chunk["section"]
        if sec not in seen_sections:
            seen_sections.add(sec)
            sections.append(sec)

    update_ancestor_summaries(doc_node_id)

    return IngestResult(
        node_id=doc_node_id,
        name=filename,
        num_chunks=len(chunks),
        sections=sections,
        parent_node_id=parent_id,
        is_new_branch=is_new_branch,
    )


# ---------------------------------------------------------------------------
# Web ingestion
# ---------------------------------------------------------------------------

def ingest_url(
    url: str,
    source_name: str | None = None,
    confirm_callback: ConfirmCallback | None = None,
    force: bool = False,
) -> IngestResult:
    """Fetch a web page, extract content, embed, and store in the DAG."""
    store.init_db()

    # Fast URL-based dedup
    existing = store.find_node_by_source_url(url)
    if existing:
        if force:
            _force_delete_by_url(url)
        else:
            return IngestResult(
                node_id=-1, name=existing["name"], num_chunks=0, sections=[],
            )

    chunks, metadata = web_fetcher.process_url(url, source_name=source_name)
    title = metadata.get("title", "Untitled")
    name = f"{source_name}: {title}" if source_name else title

    chunk_texts = [c["text"] for c in chunks]
    full_text = "\n\n".join(chunk_texts[:5])

    doc_summary = generate_leaf_description(full_text)
    description = f"Source: {url}\n\n{doc_summary}"
    summary_vector = embedder.embed_single_vector([doc_summary])[0]

    if not force and _check_duplicates(doc_summary, summary_vector):
        return IngestResult(node_id=-1, name=name, num_chunks=0, sections=[])

    parent_id = _find_best_parent(doc_summary, summary_vector, confirm_callback)

    doc_node_id = store.insert_node(
        node_type="document",
        name=name,
        description=description,
        embedding_type="single",
    )
    store.save_vectors(doc_node_id, summary_vector)

    is_new_branch = False
    if parent_id is not None:
        store.insert_edge(parent_id, doc_node_id)
        parent = store.get_node(parent_id)
        if parent and parent["node_type"] == "category":
            is_new_branch = True

    embeddings = embedder.embed_documents(chunk_texts)

    seen_sections: set[str] = set()
    sections: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_node_id = store.insert_node(
            node_type="chunk",
            name=f"{chunk['section']} (chunk {chunk['chunk_index']})",
            text=chunk["text"],
            embedding_type="multi",
            token_count=chunk["token_count"],
        )
        store.save_vectors(chunk_node_id, embeddings[i])
        store.insert_edge(doc_node_id, chunk_node_id, sort_order=chunk["chunk_index"])

        sec = chunk["section"]
        if sec not in seen_sections:
            seen_sections.add(sec)
            sections.append(sec)

    update_ancestor_summaries(doc_node_id)

    return IngestResult(
        node_id=doc_node_id,
        name=name,
        num_chunks=len(chunks),
        sections=sections,
        parent_node_id=parent_id,
        is_new_branch=is_new_branch,
    )


def ingest_feed(
    feed_url: str,
    feed_name: str | None = None,
    max_articles: int | None = None,
    confirm_callback: ConfirmCallback | None = None,
    force: bool = False,
) -> list[IngestResult]:
    """Fetch all articles from an RSS feed and ingest them."""
    store.init_db()

    articles = web_fetcher.process_feed(feed_url, feed_name=feed_name, max_articles=max_articles)
    results: list[IngestResult] = []

    for i, (chunks, metadata) in enumerate(articles):
        url = metadata.get("url", "")
        title = metadata.get("title", "Untitled")
        source = metadata.get("source_name", "")
        print(f"  [{i+1}/{len(articles)}] Ingesting: {title}")

        # URL dedup
        existing = store.find_node_by_source_url(url)
        if existing:
            if force:
                _force_delete_by_url(url)
                print(f"    Replacing existing version")
            else:
                print(f"    Skipped (already ingested)")
                results.append(IngestResult(
                    node_id=-1, name=existing["name"], num_chunks=0, sections=[],
                ))
                continue

        name = f"{source}: {title}" if source else title
        chunk_texts = [c["text"] for c in chunks]
        full_text = "\n\n".join(chunk_texts[:5])

        try:
            doc_summary = generate_leaf_description(full_text)
            description = f"Source: {url}\n\n{doc_summary}"
            summary_vector = embedder.embed_single_vector([doc_summary])[0]

            if not force and _check_duplicates(doc_summary, summary_vector):
                print(f"    Skipped (duplicate content)")
                results.append(IngestResult(node_id=-1, name=name, num_chunks=0, sections=[]))
                continue

            parent_id = _find_best_parent(doc_summary, summary_vector, confirm_callback=None)

            doc_node_id = store.insert_node(
                node_type="document",
                name=name,
                description=description,
                embedding_type="single",
            )
            store.save_vectors(doc_node_id, summary_vector)

            is_new_branch = False
            if parent_id is not None:
                store.insert_edge(parent_id, doc_node_id)
                parent = store.get_node(parent_id)
                if parent and parent["node_type"] == "category":
                    is_new_branch = True

            embeddings = embedder.embed_documents(chunk_texts)

            seen_sections: set[str] = set()
            sections: list[str] = []
            for j, chunk in enumerate(chunks):
                chunk_node_id = store.insert_node(
                    node_type="chunk",
                    name=f"{chunk['section']} (chunk {chunk['chunk_index']})",
                    text=chunk["text"],
                    embedding_type="multi",
                    token_count=chunk["token_count"],
                )
                store.save_vectors(chunk_node_id, embeddings[j])
                store.insert_edge(doc_node_id, chunk_node_id, sort_order=chunk["chunk_index"])

                sec = chunk["section"]
                if sec not in seen_sections:
                    seen_sections.add(sec)
                    sections.append(sec)

            update_ancestor_summaries(doc_node_id)

            results.append(IngestResult(
                node_id=doc_node_id,
                name=name,
                num_chunks=len(chunks),
                sections=sections,
                parent_node_id=parent_id,
                is_new_branch=is_new_branch,
            ))
        except Exception as e:
            print(f"    Error: {e}")
            results.append(IngestResult(node_id=-1, name=name, num_chunks=0, sections=[]))

    return results


def ingest_all_feeds(
    feeds_file: str,
    max_articles: int | None = None,
    confirm_callback: ConfirmCallback | None = None,
    force: bool = False,
) -> list[IngestResult]:
    """Load a feeds JSON file and ingest all feeds."""
    feeds = web_fetcher.load_feeds_file(feeds_file)
    all_results: list[IngestResult] = []

    for feed in feeds:
        feed_url = feed["url"]
        feed_name = feed.get("name")
        print(f"\nProcessing feed: {feed_name or feed_url}")
        results = ingest_feed(
            feed_url,
            feed_name=feed_name,
            max_articles=max_articles,
            confirm_callback=confirm_callback,
            force=force,
        )
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Summary propagation
# ---------------------------------------------------------------------------

def update_ancestor_summaries(node_id: int) -> None:
    """Regenerate summaries for all ancestors of a node, bottom-up."""
    ancestors = store.get_ancestors(node_id)
    # Process bottom-up (reverse order since ancestors are returned root-first)
    for ancestor in reversed(ancestors):
        children = store.get_children(ancestor["id"])
        if not children:
            continue

        child_descriptions = []
        for child in children:
            desc = child.get("description") or child.get("name", "")
            child_descriptions.append(desc)

        new_summary = generate_node_summary(ancestor["name"], child_descriptions)
        store.update_node(ancestor["id"], description=new_summary)

        # Re-embed the updated summary
        vec = embedder.embed_single_vector([new_summary])[0]
        store.save_vectors(ancestor["id"], vec)
        store.update_node(ancestor["id"], embedding_type="single")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_graph(node_id: int | None = None) -> int:
    """Generate summaries and embeddings for nodes that don't have them.

    Bottom-up: describe leaves first, then summarize parents from children.
    Returns the number of nodes enriched.
    """
    store.init_db()
    count = 0

    if node_id:
        nodes = [store.get_node(node_id)] + store.get_descendants(node_id) if store.get_node(node_id) else []
    else:
        nodes = store.get_all_nodes()

    # Enrich leaves first (generate descriptions from their text)
    for node in nodes:
        if node is None:
            continue
        children = store.get_children(node["id"])
        if children:
            continue  # not a leaf
        if node.get("description"):
            continue  # already has description
        if not node.get("text"):
            continue  # no text to describe

        desc = generate_leaf_description(node["text"])
        store.update_node(node["id"], description=desc)
        count += 1

    # Enrich branch nodes bottom-up (nodes with children but no description)
    # Sort by depth descending so we process deepest first
    branch_nodes = []
    for node in nodes:
        if node is None:
            continue
        children = store.get_children(node["id"])
        if not children:
            continue
        branch_nodes.append(node)

    # Simple heuristic: process nodes with fewer ancestors first (deeper nodes)
    branch_nodes.sort(key=lambda n: len(store.get_ancestors(n["id"])), reverse=True)

    for node in branch_nodes:
        children = store.get_children(node["id"])
        child_descs = [c.get("description") or c.get("name", "") for c in children]
        summary = generate_node_summary(node["name"], child_descs)
        store.update_node(node["id"], description=summary)

        # Embed the summary as single vector
        vec = embedder.embed_single_vector([summary])[0]
        store.save_vectors(node["id"], vec)
        store.update_node(node["id"], embedding_type="single")
        count += 1

    return count


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_documents(
    query: str,
    top_k: int = 5,
    flat: bool = False,
    node_id: int | None = None,
    verbose: bool = False,
) -> tuple[list[SearchResult], list[dict]] | list[SearchResult]:
    """Search ingested documents and return ranked results.

    Args:
        query: Natural-language query string.
        top_k: Number of results to return.
        flat: If True, use flat search (no graph navigation).
        node_id: If set with flat search, restrict to leaves under this node.
        verbose: If True, return (results, trace) with traversal info.
    """
    store.init_db()

    trace: list[dict] = []
    if flat:
        raw_results = searcher.flat_search(query, top_k=top_k, node_id=node_id)
    elif verbose:
        raw_results, trace = searcher.graph_search(query, top_k=top_k, verbose=True)
    else:
        raw_results = searcher.graph_search(query, top_k=top_k)

    results = [
        SearchResult(
            node_id=r["node_id"],
            score=r["score"],
            text=r.get("text", ""),
            name=r.get("name", ""),
            node_type=r.get("node_type", ""),
            highlights=r.get("highlights", []),
        )
        for r in raw_results
    ]

    return (results, trace) if verbose else results


# ---------------------------------------------------------------------------
# Graph listing
# ---------------------------------------------------------------------------

def list_nodes() -> list[NodeInfo]:
    """Return metadata for all root-level nodes and their immediate structure."""
    store.init_db()

    nodes = store.get_all_nodes()
    return [
        NodeInfo(
            node_id=n["id"],
            name=n["name"],
            node_type=n["node_type"],
            description=n.get("description", ""),
            child_count=store.count_children(n["id"]),
            embedding_type=n.get("embedding_type"),
        )
        for n in nodes
    ]


def delete_node(node_id: int) -> bool:
    """Delete a node and all descendants. Returns True if the node existed."""
    store.init_db()

    node = store.get_node(node_id)
    if node is None:
        return False

    store.delete_node(node_id)
    return True
