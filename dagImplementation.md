# DAG-Based Hierarchical Embedding Search System

## Context

The current system is a flat document->chunk ColBERT search pipeline. The goal is to transform it into a hierarchical DAG where embeddings exist at every level, enabling agentic search navigation — an LLM narrows down through layers (domain -> source -> document -> chunk) rather than searching everything at once. This saves context tokens and mirrors how humans navigate information.

## Design Decisions (Locked In)

1. Unified `nodes` + `edges` tables (not table-per-level)
2. Auto-generated LLM summaries at non-leaf levels, updated when children change
3. Score threshold routing with min/max bounds, LLM fallback when ambiguous
4. Mixed embeddings: single-vector (Jina v3) for upper nodes, ColBERT multi-vector for leaves
5. Fully-inferred ingestion — LLM examines content + existing graph, routes autonomously or interactively
6. Autonomous mode as opt-in flag (default is interactive for ambiguous cases)
7. Dedup: embedding similarity fast-pass, then LLM confirms

---

## Phase 0: Config & Dependencies

**File: `config.py`**

Add:
- `JINA_SINGLE_VECTOR_URL = "https://api.jina.ai/v1/embeddings"`
- `JINA_SINGLE_VECTOR_MODEL = "jina-embeddings-v3"`
- `JINA_SINGLE_VECTOR_DIMENSIONS = 1024`
- Threshold constants: `SEARCH_SCORE_THRESHOLD_HIGH = 0.75`, `SEARCH_SCORE_THRESHOLD_LOW = 0.40`, `SEARCH_MAX_CANDIDATES_PER_LEVEL = 5`, `SEARCH_LLM_AMBIGUITY_RANGE = 0.15`
- `DEDUP_SIMILARITY_THRESHOLD = 0.92`
- `CLAUDE_CLI_PATH = "claude"` — path to Claude Code CLI binary

**File: `requirements.txt`** — no new deps needed (claude CLI called via subprocess)

---

## Phase 1: Schema Migration (`core/store.py`)

**New schema:**

```sql
CREATE TABLE nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type       TEXT NOT NULL,        -- 'root','category','document','section','chunk'
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',      -- LLM-generated summary (branch nodes)
    text            TEXT DEFAULT '',      -- full text (leaf nodes)
    embedding_type  TEXT DEFAULT NULL,    -- 'single' | 'multi' | NULL
    token_count     INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE edges (
    parent_id   INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    child_id    INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    sort_order  INTEGER DEFAULT 0,
    PRIMARY KEY (parent_id, child_id)
);
```

**Migration from v1:** `migrate_v1_to_v2()` — maps documents to document nodes, chunks to chunk nodes, creates edges, renames vector files from old chunk IDs to new node IDs. Copy-then-delete for safety.

**New store API:**
- `insert_node()`, `get_node()`, `update_node()`, `delete_node()` (cascade)
- `insert_edge()`, `delete_edge()`, `get_children()`, `get_parents()`, `get_roots()`
- `get_descendants()`, `get_ancestors()`, `get_leaf_nodes()`
- Vector I/O unchanged (keyed on node_id instead of chunk_id)

---

## Phase 2: Dual Embedding Support (`core/embedder.py`)

Add single-vector embedding alongside existing multi-vector:

- Extract retry logic into shared `_api_call_with_retry()` helper
- `embed_single_vector(texts, input_type) -> list[np.ndarray]` — calls Jina v3 `/v1/embeddings`, returns 1-D arrays shape `(1024,)`
- `embed_single_query(query) -> np.ndarray` — single-vector query embedding
- Existing `embed_documents()` and `embed_query()` unchanged

Both 1-D and 2-D vectors store fine in `.npz`. The `embedding_type` field on the node tells the searcher which scoring function to use.

---

## Phase 3: LLM Integration (`core/llm.py` — new file)

Uses Claude Code CLI (`claude -p`) via subprocess for all LLM calls. No SDK dependency — leverages user's existing Claude Code subscription. All calls request JSON output via prompt instructions.

**Functions:**
- `generate_node_summary(node_name, children_descriptions) -> str` — 1-2 sentence summary for branch nodes
- `generate_leaf_description(text) -> str` — short description of chunk content
- `propose_placement(content_summary, existing_graph, candidates) -> dict` — returns action (attach/create_new/duplicate), target node, confidence, reasoning
- `llm_route_search(query, candidates_with_scores) -> list[int]` — pick which nodes to descend into when scores are ambiguous
- `confirm_duplicate(new_summary, existing_summary, score) -> dict` — is_duplicate, action (skip/merge/keep_both)

---

## Phase 4: Graph Search (`core/searcher.py` + `core/graph.py`)

**New file `core/graph.py`:** graph traversal utilities — `get_nodes_at_level()`, `get_graph_description()`, `find_leaf_descendants()`, `compute_depth()`

**Searcher additions:**
- `cosine_score(query_vec, node_vec) -> float` — for single-vector nodes
- `score_node(query_vecs, node_id, embedding_type) -> float` — dispatch to cosine or MaxSim
- `graph_search(query, top_k) -> list[dict]` — the main new entry point:
  1. Embed query as both single-vector and multi-vector
  2. Start at roots, score with cosine similarity
  3. Threshold routing: >HIGH take top N descend, <LOW stop, ambiguous -> LLM
  4. Repeat at each level until reaching leaves
  5. Score leaves with MaxSim, return top-k with highlights

**Preserve `flat_search()`** (renamed from current `search()`) as fallback for unmigrated data or explicit `--flat` flag.

---

## Phase 5: Ingestion Pipeline (`service.py`)

Rewrite `ingest_document()` with graph-aware flow:

1. Parse and chunk document (existing chunker)
2. LLM generates document summary
3. Embed summary as single vector
4. **Dedup check:** cosine sim against existing document nodes, LLM confirms if above threshold
5. **Route placement:** score against graph nodes, LLM proposes placement
6. If ambiguous + interactive: call `confirm_callback` (CLI prompts user). If autonomous: accept LLM recommendation
7. Create document node, attach to chosen parent (or create new branch node)
8. Create leaf nodes with multi-vector embeddings, attach as children
9. Update ancestor summaries bottom-up

**Callback pattern for interactive/autonomous:**
```python
def ingest_document(file_path, confirm_callback=None):
    # confirm_callback=None means autonomous
    # confirm_callback=some_function means interactive
```

**`update_ancestor_summaries(node_id)`** — walk to roots, regenerate summaries, re-embed as single vectors.

---

## Phase 6: CLI Updates (`main.py`)

Updated commands:
- `ingest <file> [--autonomous]`
- `ingest-ticker <ticker> [--autonomous]`
- `search <query> [--top-k N] [--flat]`
- `list [--flat]` — default shows tree view
- `delete <node_id>`
- `graph [--depth N]` — show graph structure
- `migrate` — explicitly run v1->v2 migration
- `enrich [--node ID]` — generate summaries/embeddings for branch nodes missing them

---

## Phase 7: Post-Migration Enrichment

`enrich_graph()` — bottom-up pass: generate descriptions for leaves from their text, then walk up generating summaries for parents. Embed each as single vector. Separate phase because it requires LLM API calls (cost) and can run incrementally.

---

## Implementation Order & Parallelism

```
Phase 0 (config)     -- first, no deps
Phase 1 (store)    \
Phase 2 (embedder)  |-- parallel, different files
Phase 3 (LLM)      /
Phase 4 (searcher)   -- depends on 1+2+3
Phase 5 (service)    -- depends on 4
Phase 6 (CLI)        -- depends on 5
Phase 7 (enrich)     -- depends on 6
```

## Key Risks

- **Migration data loss:** vector file rename — mitigate with copy-then-delete
- **LLM cost/latency:** every ingestion needs Claude calls — mitigate with fallback to embedding-only routing if no API key
- **LLM output parsing:** JSON from Claude can be malformed — wrap in try/except with fallbacks
- **Graph depth creep:** cap at ~5 levels, warn during ingestion

## Verification

1. Existing tests pass after Phase 1 migration
2. `python main.py migrate` converts old data correctly
3. `python main.py enrich` populates summaries on migrated branch nodes
4. `python main.py ingest <file>` places document in graph with LLM routing
5. `python main.py search <query>` navigates graph level-by-level, returns highlighted results
6. `python main.py search <query> --flat` still works as before
7. `python main.py graph` shows the full DAG structure
8. Run full test suite: `python -m pytest tests/ -v`

## Critical Files

- `config.py` — new settings
- `core/store.py` — complete rewrite (nodes/edges + migration)
- `core/embedder.py` — add single-vector support
- `core/searcher.py` — add cosine scoring + graph traversal search
- `core/graph.py` — new: graph traversal utilities
- `core/llm.py` — new: Claude API integration
- `service.py` — rewrite ingestion + search orchestration
- `main.py` — updated CLI
