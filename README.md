# ColBERT Multi-Vector Search Pipeline

A document search pipeline using [Jina ColBERT-v2](https://jina.ai) multi-vector embeddings. Unlike traditional single-vector search, ColBERT produces one embedding per token, enabling precise term-level matching with MaxSim scoring.

Built for financial documents (SEC 10-K filings), but works with any markdown.

## How it works

1. **Ingest** -- parse markdown into sections (split at the 8192-token API limit if needed), embed each section via Jina ColBERT-v2 (one 128-d vector per token)
2. **Store** -- metadata in SQLite, token vectors in `.npz` files
3. **Search** -- embed query, score every chunk with MaxSim, return top-k with per-word highlighting that shows which query term matched

### MaxSim scoring

For each query token, find its highest cosine similarity against all document tokens, then sum those maxima. This lets each query term independently find its best match in the document.

```
sim = normalize(Q) @ normalize(D).T    # (Q_tokens, D_tokens)
score = sum(max(sim, axis=1))           # one max per query token, then sum
```

### Token-level highlighting

Search results highlight which words matched and **which query term they matched against**:

```
The company reported strong revenue growth in Q4, driven by
increased *demand*[growth] across all segments. Total
>>>earnings<<<[revenue] exceeded *expectations*[growth] for
the fiscal year.
```

- `>>>word<<<` / bold red = strong match (score >= 0.80)
- `*word*` / bold yellow = good match (score >= 0.72)
- `[query_term]` = the query word that triggered the match

This is done by extracting `argmax` from the similarity matrix to identify which query token best matched each document token, then mapping token indices back to words via the XLM-RoBERTa tokenizer.

**Tokenizer alignment:** Jina ColBERT-v2 uses XLM-RoBERTa as its backbone. The API prepends a marker token (`[D]` for documents, `[Q]` for queries) before the tokenizer output. Document vectors have 1 extra token vs local tokenization; query vectors are padded to a fixed 32 tokens. The code accounts for both offsets when mapping API vector indices back to source words.

## Setup

```bash
pip install -r requirements.txt
```

Put your Jina API key in `apiKey.txt` in the project root.

### Dependencies

- `numpy` -- vector math
- `requests` -- Jina API calls
- `tokenizers` -- XLM-RoBERTa tokenizer for token-to-word mapping
- `edgartools` -- SEC EDGAR filing fetcher

## CLI usage

### Ingest a markdown file

```bash
python main.py ingest path/to/document.md
```

### Ingest an SEC 10-K filing by ticker

```bash
python main.py ingest-ticker AAPL
```

Fetches the latest 10-K from SEC EDGAR, extracts Items 1-9B, embeds, and stores.

### Search

```bash
python main.py search "revenue growth drivers"
python main.py search "risk factors" --top-k 10
python main.py search "operating expenses" --section "Item 7"
python main.py search "debt obligations" --document-id 1
```

### List ingested documents

```bash
python main.py list
```

### Delete a document

```bash
python main.py delete 3
```

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `JINA_MODEL` | `jina-colbert-v2` | Embedding model |
| `JINA_DIMENSIONS` | `128` | Vectors per token |
| `DEFAULT_TOP_K` | `5` | Search results returned |

## Project structure

```
embedded/
  main.py              CLI entry point
  service.py           Orchestration layer (ingest, search, list, delete)
  config.py            Paths, API settings, search params

  core/
    embedder.py        Jina ColBERT-v2 API client (batched, with retry)
    searcher.py        MaxSim scoring, token relevance, word highlighting
    chunker.py         Markdown section parsing and token-bounded chunking
    store.py           SQLite metadata + NumPy vector I/O
    edgar_fetcher.py   SEC EDGAR 10-K fetcher via edgartools

  data/
    embeddings.db      SQLite (documents + chunks tables)
    vectors/           {chunk_id}.npz files (N_tokens x 128 float32)

  tests/
    test_searcher.py
    test_service.py
    test_chunker.py
    test_embedder.py
    test_store.py
    test_edgar_fetcher.py
```

## Storage

- **SQLite** (`data/embeddings.db`) -- document and chunk metadata, section labels, token counts
- **NumPy archives** (`data/vectors/{chunk_id}.npz`) -- per-token embedding vectors, shape `(N_tokens, 128)`

Vectors are stored separately from SQLite to avoid BLOB column bloat and to leverage NumPy's native compressed I/O.

## Tests

```bash
python -m pytest tests/ -v
```
