# DAG-Based Hierarchical Embedding Search

A knowledge graph search pipeline that organizes content into a directed acyclic graph (DAG) and navigates it level-by-level using embeddings. Upper-level nodes use single-vector embeddings for fast routing; leaf nodes use [Jina ColBERT-v2](https://jina.ai) multi-vector embeddings for precise token-level matching.

The idea: instead of searching everything at once, narrow down through layers (domain -> source -> document -> chunk) the same way a human navigates information. This saves context for an LLM abstraction layer on top.

## How it works

1. **Ingest** -- parse markdown into sections, embed chunks with ColBERT multi-vector, then an LLM decides where the document fits in the graph (or creates a new branch). Document-level summaries are embedded as single vectors for fast routing.
2. **Store** -- nodes and edges in SQLite (DAG structure), token vectors in `.npz` files.
3. **Search** -- embed query, navigate the DAG level-by-level using threshold-based routing, score leaf chunks with MaxSim, return top-k with per-word highlighting.

### Graph navigation

Search starts at root nodes, scores them with cosine similarity against a single-vector query embedding, picks the best candidates via score thresholds, descends into their children, and repeats until reaching leaf nodes. Leaf nodes are scored with full MaxSim (token-level matching). An LLM fallback kicks in when scores are ambiguous.

```
Query: "mortgage rates housing affordability"

Root level (cosine sim):
  fed_rate_hike.md        → 0.82  ✓ descend
  gpt5_launch.md          → 0.31  ✗ skip
  us_israel_iran_war.md   → 0.28  ✗ skip

Document level:
  fed_rate_hike.md        → descend into chunks

Chunk level (MaxSim):
  Housing Market (chunk 0) → 21.23  ← top result
  Summary (chunk 0)        → 18.17
  ...
```

### MaxSim scoring

For each query token, find its highest cosine similarity against all document tokens, then sum those maxima. Each query term independently finds its best match in the document.

```
sim = normalize(Q) @ normalize(D).T    # (Q_tokens, D_tokens)
score = sum(max(sim, axis=1))           # one max per query token, then sum
```

### Token-level highlighting

Search results highlight which words matched and which query term they matched against:

```
The company reported strong revenue growth in Q4, driven by
increased *demand*[growth] across all segments. Total
>>>earnings<<<[revenue] exceeded *expectations*[growth] for
the fiscal year.
```

- `>>>word<<<` / bold red = strong match (score >= 0.80)
- `*word*` / bold yellow = good match (score >= 0.72)
- `[query_term]` = the query word that triggered the match

### LLM-powered ingestion

When you ingest a document, the pipeline:
1. Generates a summary via Claude CLI
2. Embeds the summary as a single vector
3. Checks for duplicates (embedding similarity + LLM confirmation)
4. Asks the LLM where to place it in the existing graph (or creates a new category)
5. Creates chunk nodes with ColBERT multi-vector embeddings
6. Updates ancestor summaries bottom-up

In **interactive mode** (default), the CLI asks for confirmation on ambiguous placement. In **autonomous mode** (`--autonomous`), the LLM makes all decisions.

## Setup

```bash
pip install -r requirements.txt
```

- Put your Jina API key in `apiKey.txt` in the project root.
- Ensure the `claude` CLI is installed and authenticated (used for LLM calls during ingestion/routing).

### Dependencies

- `numpy` -- vector math
- `requests` -- Jina API calls
- `tokenizers` -- XLM-RoBERTa tokenizer for token-to-word mapping
- `edgartools` -- SEC EDGAR filing fetcher
- `claude` CLI -- LLM calls for summarization, routing, and dedup (via subprocess)

## CLI usage

### Ingest a markdown file

```bash
python main.py ingest path/to/document.md               # interactive mode
python main.py ingest path/to/document.md --autonomous   # no confirmations
```

### Ingest an SEC 10-K filing by ticker

```bash
python main.py ingest-ticker AAPL
python main.py ingest-ticker AAPL --autonomous
```

### Search

```bash
# Graph search (navigates the DAG level-by-level)
python main.py search "revenue growth drivers"
python main.py search "risk factors" --top-k 10

# Flat search (brute-force all leaf chunks, no graph navigation)
python main.py search "revenue growth drivers" --flat
```

### Show graph structure

```bash
python main.py graph
python main.py graph --depth 2
```

### List nodes

```bash
python main.py list           # tree view
python main.py list --flat    # flat table
```

### Delete a node

```bash
python main.py delete 3       # deletes node 3 and all descendants
```

### Enrich nodes with summaries

```bash
python main.py enrich              # all nodes missing summaries
python main.py enrich --node 5     # specific subtree only
```

### Migrate from v1

```bash
python main.py migrate
```

Automatically converts old `documents`+`chunks` tables to the new `nodes`+`edges` schema. Runs on first `init_db()` call anyway, but this command lets you trigger it explicitly.

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `JINA_MODEL` | `jina-colbert-v2` | Multi-vector embedding model |
| `JINA_DIMENSIONS` | `128` | Vectors per token (ColBERT) |
| `JINA_SINGLE_VECTOR_MODEL` | `jina-embeddings-v3` | Single-vector embedding model |
| `JINA_SINGLE_VECTOR_DIMENSIONS` | `1024` | Single-vector dimensions |
| `DEFAULT_TOP_K` | `5` | Search results returned |
| `SEARCH_SCORE_THRESHOLD_HIGH` | `0.75` | Score above which to descend confidently |
| `SEARCH_SCORE_THRESHOLD_LOW` | `0.40` | Score below which to stop descending |
| `SEARCH_MAX_CANDIDATES_PER_LEVEL` | `5` | Max nodes to explore per level |
| `SEARCH_LLM_AMBIGUITY_RANGE` | `0.15` | Score range triggering LLM routing |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity for duplicate detection |

## Project structure

```
embedded/
  main.py              CLI entry point
  service.py           Orchestration (ingest, search, enrich, delete)
  config.py            Paths, API settings, thresholds

  core/
    embedder.py        Jina API client — multi-vector (ColBERT) + single-vector
    searcher.py        MaxSim + cosine scoring, flat search, graph search
    chunker.py         Markdown section parsing and token-bounded chunking
    store.py           SQLite nodes+edges DAG, NumPy vector I/O, v1 migration
    graph.py           Graph traversal utilities (description, depth, leaves)
    llm.py             Claude CLI integration (summarization, routing, dedup)
    edgar_fetcher.py   SEC EDGAR 10-K fetcher via edgartools

  data/
    embeddings.db      SQLite (nodes + edges tables)
    vectors/           {node_id}.npz files

  tests/
    test_store.py      Node/edge CRUD, graph queries, migration, vector I/O
    test_embedder.py   Multi-vector + single-vector, batching, retry
    test_llm.py        Claude CLI mocking, JSON parsing, all LLM functions
    test_graph.py      Graph description, leaf descendants, depth
    test_searcher.py   MaxSim, cosine, flat search, graph search, routing
    test_service.py    Ingestion, dedup, enrichment, search, delete
    test_chunker.py    Section parsing, token splitting
    test_edgar_fetcher.py  EDGAR API mocking
```

## Storage

- **SQLite** (`data/embeddings.db`) -- `nodes` table (id, type, name, description, text, embedding_type, token_count) and `edges` table (parent_id, child_id, sort_order) forming a DAG
- **NumPy archives** (`data/vectors/{node_id}.npz`) -- either `(N_tokens, 128)` float32 for multi-vector nodes or `(1024,)` float32 for single-vector nodes

Vectors are stored separately from SQLite to avoid BLOB column bloat and to leverage NumPy's native compressed I/O.

## Tests

```bash
python -m pytest tests/ -v
```

150 tests covering all modules.
