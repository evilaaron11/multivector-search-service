# DAG-Based Hierarchical Embedding Search

A knowledge graph search pipeline that organizes content into a directed acyclic graph (DAG) and navigates it level-by-level using embeddings. Upper-level nodes use single-vector embeddings for fast routing; leaf nodes use [Jina ColBERT-v2](https://jina.ai) multi-vector embeddings for precise token-level matching.

The idea: instead of searching everything at once, narrow down through layers (source -> document -> chunk) the same way a human navigates information. RSS feed ingestion automatically groups articles under their source (e.g. "BBC Middle East", "Al Jazeera") with publication dates attached to every node. This saves context for an LLM abstraction layer on top.

## How it works

1. **Ingest** -- parse markdown into sections, embed chunks with ColBERT multi-vector, then an LLM decides where the document fits in the graph (or creates a new branch). Document-level summaries are embedded as single vectors for fast routing.
2. **Store** -- nodes and edges in SQLite (DAG structure), token vectors in `.npz` files.
3. **Search** -- embed query, navigate the DAG level-by-level using threshold-based routing, score leaf chunks with MaxSim, return top-k with per-word highlighting.

### Graph navigation

Search starts at root nodes, scores them with cosine similarity against a single-vector query embedding, picks the best candidates via score thresholds, descends into their children, and repeats until reaching leaf nodes. Leaf nodes are scored with full MaxSim (token-level matching). An LLM fallback kicks in when scores are ambiguous.

```
Query: "Iran war Strait of Hormuz shipping"

Source level (cosine sim):
  BBC Middle East           → 0.68  ✓ descend
  The Guardian Middle East  → 0.61  ✓ descend
  Ars Technica              → 0.38  ✗ skip
  TechCrunch                → 0.29  ✗ skip

Document level:
  Iran-backed Houthis join war [2026-03-28]       → 0.67  ✓ descend
  Houthi threat to Red Sea shipping [2026-03-28]  → 0.63  ✓ descend
  Funeral for journalists [2026-03-29]            → 0.46  ✗ skip

Chunk level (MaxSim):
  Houthis join war (chunk 0)         → 22.30  ← top result
  Red Sea shipping threat (chunk 0)  → 22.11
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
- The CLI path defaults to `claude` (expects it on `$PATH`). If your environment installs it elsewhere (e.g. `~/.claude/local/claude` on WSL, or a VS Code extension path), set the `CLAUDE_CLI_PATH` environment variable:
  ```bash
  export CLAUDE_CLI_PATH=/home/you/.claude/local/claude
  ```

### Dependencies

- `numpy` -- vector math
- `requests` -- Jina API calls, RSS feed fetching
- `trafilatura` -- web article content extraction
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

### Ingest a single RSS feed

```bash
python main.py ingest-feed https://feeds.bbci.co.uk/news/world/middle_east/rss.xml --name "BBC Middle East"
python main.py ingest-feed https://www.aljazeera.com/xml/rss/all.xml --name "Al Jazeera" --max-articles 5
```

Articles are placed under a **source category node** named after the feed, following the hierarchy: `source -> document -> chunk`. Each document node includes the article's publication date (e.g. `[2026-03-29]`) and each chunk is prefixed with the date so search results and LLM context include temporal information.

### Ingest all feeds from a file

```bash
python main.py ingest-feeds feeds.json
python main.py ingest-feeds feeds.json --max-articles 3
python main.py ingest-feeds feeds.json --force   # re-ingest all
```

The feeds file is a JSON array of `{"url": "...", "name": "..."}` objects (see `feeds.json` for an example). Each feed gets its own source category node.

### Ingest a web page by URL

```bash
python main.py ingest-url https://example.com/article
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
| `CLAUDE_CLI_PATH` | `claude` | Path to Claude CLI binary (override via env var) |
| `RSS_MAX_ARTICLES_PER_FEED` | `20` | Default max articles per feed |
| `RSS_MIN_ARTICLE_WORDS` | `100` | Skip articles shorter than this |
| `RSS_FETCH_DELAY` | `1.0` | Seconds between article fetches (rate limiting) |

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
    web_fetcher.py     RSS/Atom feed parser, web article fetcher via trafilatura

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
    test_web_fetcher.py    RSS parsing, feed processing, article extraction
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
