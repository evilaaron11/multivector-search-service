# Multi-Vector Embedding Pipeline - Implementation Plan

## Overview
SEC 10-Q/10-K document search using Jina ColBERT-v2 multi-vector embeddings.
SQLite for metadata, NumPy (.npz) for vector storage, brute-force MaxSim retrieval.
Service layer designed for future FastAPI and local LLM integration.

## Architecture
```
main.py  -->  service.py  -->  core/chunker.py    (MD parsing + chunking)
(CLI)         (orchestration)   core/embedder.py   (Jina ColBERT-v2 API)
                                core/store.py      (SQLite + .npz vectors)
                                core/searcher.py   (MaxSim scoring)
              config.py         (settings, paths, API key)
```

## How Search Works (MaxSim)
Each chunk is stored as a matrix of per-token vectors (N_tokens x 128).
At query time, each query token finds its best match across all document tokens.
Scores are summed across query tokens to produce a final relevance score.
This preserves term-level precision -- important for financial terminology.

---

## Phase 1: Project Setup & Config
- [ ] Create directory structure (core/, data/, data/vectors/)
- [ ] Create requirements.txt (numpy, requests)
- [ ] Create config.py (API key loading, paths, constants)
- [ ] Create core/__init__.py

## Phase 2: Database & Storage Layer
- [ ] Create core/store.py
  - SQLite schema: documents table (id, filename, ingested_at)
  - SQLite schema: chunks table (id, document_id, text, section, chunk_index, token_count)
  - Insert/query helpers for documents and chunks
  - Save/load NumPy embedding matrices to data/vectors/{chunk_id}.npz
  - List documents, get chunks by document, get all chunk IDs

## Phase 3: Markdown Ingestion & Chunking
- [ ] Create core/chunker.py
  - Parse .md file into sections by header (# and ##)
  - Split oversized sections into chunks (target ~4000 tokens, hard max 8192)
  - Overlap between chunks (~200 tokens) to preserve context at boundaries
  - Return list of chunks with metadata (section name, chunk index, token count)
  - Token counting via simple whitespace split (good enough for POC, swap later)

## Phase 4: Embedding via Jina API
- [ ] Create core/embedder.py
  - Call Jina ColBERT-v2 API (POST /v1/multi-vector)
  - Handle document embedding (input_type: "document")
  - Handle query embedding (input_type: "query")
  - Batch inputs to respect API rate limits
  - Return numpy arrays (N_tokens x 128 dims)
  - Error handling and retries

## Phase 5: Search (MaxSim Retrieval)
- [ ] Create core/searcher.py
  - Load query embedding (2D array: Q_tokens x 128)
  - Load candidate document embeddings from .npz files
  - MaxSim: for each query token, max cosine sim across all doc tokens, then sum
  - Return top-k results sorted by score
  - Optional metadata filters (section, document_id) via SQLite pre-filtering

## Phase 6: Service Layer
- [ ] Create service.py
  - ingest_document(file_path) -> parse, chunk, embed, store. Returns IngestResult
  - search(query, top_k, filters) -> embed query, score, return ranked results
  - list_documents() -> all ingested documents with chunk counts
  - delete_document(document_id) -> remove doc, chunks, and vector files
  - All functions return typed dataclasses (IngestResult, SearchResult, DocumentInfo)

## Phase 7: CLI
- [ ] Create main.py
  - python main.py ingest <file.md>
  - python main.py search "query" [--top-k 5] [--section "Item 7"]
  - python main.py list
  - python main.py delete <document_id>
  - Pretty-print results with scores, section names, and text snippets

## Phase 8: Testing & Validation
- [ ] Create a sample 10-K markdown file for testing
- [ ] Verify ingestion pipeline end-to-end
- [ ] Verify search returns sensible results
- [ ] Test edge cases: empty sections, very long sections, special characters

---

## Key Constants
- Embedding model: jina-colbert-v2
- Dimensions: 128 per token
- Max input tokens: 8192 (model hard limit)
- Target chunk size: ~4000 tokens (balances precision vs LLM context budget)
- Chunk overlap: ~200 tokens (preserves boundary context)
- API endpoint: https://api.jina.ai/v1/multi-vector

## Future Phases (not for today)
- FastAPI wrapper around service.py (/ingest, /search, /documents endpoints)
- Local LLM integration (Ollama, llama.cpp, or similar) for RAG:
  - Retrieve top-k chunks via MaxSim
  - Feed retrieved context + query to local LLM for answer generation
  - Model-agnostic interface so any local model can be swapped in
  - Streaming responses for interactive use
- Weaviate or Qdrant for production vector storage (replace brute-force MaxSim)
- EDGAR API integration for automatic filing download + HTML-to-MD conversion
- Hybrid search (BM25 keyword + MaxSim multi-vector)
- ColPali/ColQwen for table-heavy pages (visual embeddings)
- Metadata enrichment: ticker normalization, fiscal period parsing from XBRL
