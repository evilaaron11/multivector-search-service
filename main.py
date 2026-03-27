"""CLI entry point for the multi-vector embedding pipeline.

Phase 7: provides ``ingest``, ``search``, ``list``, and ``delete``
subcommands backed by the service orchestration layer.
"""

from __future__ import annotations

import argparse
import sys

# Ensure Unicode output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import service


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _print_ingest(result: service.IngestResult) -> None:
    print(f"Ingested: {result.filename}")
    print(f"  Document ID : {result.document_id}")
    print(f"  Chunks      : {result.num_chunks}")
    print(f"  Sections    : {', '.join(result.sections)}")


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "were", "be",
    "been", "has", "had", "have", "that", "this", "not", "are", "as", "—",
    "he", "she", "we", "they", "if", "so", "no", "do", "did", "than",
})


def _is_content_word(word: str) -> bool:
    """Check if a word is meaningful content (not a stopword or punctuation)."""
    clean = word.strip(".,;:!?\"'()-–—*[]{}|/").lower()
    return len(clean) > 2 and clean not in _STOPWORDS


def _highlight_text(highlights: list[tuple[str, float, str]], threshold: float = 0.72) -> str:
    """Render text with ANSI highlighting for high-relevance words.

    Words scoring above threshold get bold yellow with matched query term.
    Words scoring above a higher threshold get bold red (strongest match).
    Stopwords and short words are never highlighted.
    """
    if not highlights:
        return ""

    BOLD_YELLOW = "\033[1;33m"
    BOLD_RED = "\033[1;31m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    high_threshold = 0.80
    parts = []
    for word, score, matched in highlights:
        if not _is_content_word(word):
            parts.append(word)
        elif score >= high_threshold:
            tag = f"{DIM}[{matched}]{RESET}" if matched else ""
            parts.append(f"{BOLD_RED}{UNDERLINE}{word}{RESET}{tag}")
        elif score >= threshold:
            tag = f"{DIM}[{matched}]{RESET}" if matched else ""
            parts.append(f"{BOLD_YELLOW}{word}{RESET}{tag}")
        else:
            parts.append(word)
    return " ".join(parts)


def _highlight_text_plain(highlights: list[tuple[str, float, str]], threshold: float = 0.72) -> str:
    """Render text with bracket markers for non-ANSI terminals."""
    if not highlights:
        return ""

    high_threshold = 0.80
    parts = []
    for word, score, matched in highlights:
        if not _is_content_word(word):
            parts.append(word)
        elif score >= high_threshold:
            tag = f"[{matched}]" if matched else ""
            parts.append(f">>>{word}<<<{tag}")
        elif score >= threshold:
            tag = f"[{matched}]" if matched else ""
            parts.append(f"*{word}*{tag}")
        else:
            parts.append(word)
    return " ".join(parts)


def _print_search(results: list[service.SearchResult]) -> None:
    if not results:
        print("No results found.")
        return

    sep = "━" * 60

    for rank, r in enumerate(results, start=1):
        print(sep)
        print(f"[{rank}] Score: {r.score:.4f}")
        print(f"    Section:  {r.section}")
        print(f"    Document: Doc {r.document_id}, Chunk {r.chunk_index}")
        print()

        if r.highlights:
            highlighted = _highlight_text(r.highlights)
            # Wrap to ~80 chars per line with indent
            words = highlighted.split(" ")
            lines = []
            current_line = "    "
            for w in words:
                # Strip ANSI for length check
                clean = w.replace("\033[1;33m", "").replace("\033[1;31m", "").replace("\033[4m", "").replace("\033[2m", "").replace("\033[0m", "")
                if len(current_line) + len(clean) + 1 > 80 and current_line.strip():
                    lines.append(current_line)
                    current_line = "    " + w
                else:
                    current_line += " " + w if current_line.strip() else "    " + w
            if current_line.strip():
                lines.append(current_line)
            print("\n".join(lines))
        else:
            snippet = r.text[:800]
            if len(r.text) > 800:
                snippet += " ..."
            indented = "\n    ".join(snippet.split("\n"))
            print(f"    {indented}")

        print()

    print(sep)
    print(f"\n\033[1;31m■\033[0m strong match  \033[1;33m■\033[0m good match")


def _print_list(docs: list[service.DocumentInfo]) -> None:
    if not docs:
        print("No documents ingested yet.")
        return

    # Column widths
    id_w = max(len("ID"), max(len(str(d.document_id)) for d in docs))
    fn_w = max(len("Filename"), max(len(d.filename) for d in docs))
    cc_w = max(len("Chunks"), max(len(str(d.chunk_count)) for d in docs))
    dt_w = max(len("Ingested At"), max(len(d.ingested_at) for d in docs))

    header = (f"{'ID':>{id_w}}  {'Filename':<{fn_w}}  "
              f"{'Chunks':>{cc_w}}  {'Ingested At':<{dt_w}}")
    print(header)
    print("-" * len(header))

    for d in docs:
        print(f"{d.document_id:>{id_w}}  {d.filename:<{fn_w}}  "
              f"{d.chunk_count:>{cc_w}}  {d.ingested_at:<{dt_w}}")


def _print_delete(document_id: int, deleted: bool) -> None:
    if deleted:
        print(f"Document {document_id} deleted successfully.")
    else:
        print(f"Document {document_id} not found.")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    result = service.ingest_document(args.file_path)
    _print_ingest(result)


def cmd_ingest_ticker(args: argparse.Namespace) -> None:
    result = service.ingest_ticker(args.ticker)
    _print_ingest(result)


def cmd_search(args: argparse.Namespace) -> None:
    results = service.search_documents(
        query=args.query,
        top_k=args.top_k,
        document_id=args.document_id,
        section=args.section,
    )
    _print_search(results)


def cmd_list(args: argparse.Namespace) -> None:
    docs = service.list_documents()
    _print_list(docs)


def cmd_delete(args: argparse.Namespace) -> None:
    deleted = service.delete_document(args.document_id)
    _print_delete(args.document_id, deleted)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-vector embedding pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a markdown document")
    p_ingest.add_argument("file_path", help="Path to the markdown file")
    p_ingest.set_defaults(func=cmd_ingest)

    # ingest-ticker
    p_ingest_ticker = subparsers.add_parser(
        "ingest-ticker", help="Ingest an SEC filing by stock ticker",
    )
    p_ingest_ticker.add_argument(
        "ticker", help="Stock ticker symbol (e.g., AAPL, INTC)",
    )
    p_ingest_ticker.set_defaults(func=cmd_ingest_ticker)

    # search
    p_search = subparsers.add_parser("search", help="Search ingested documents")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=5,
                          help="Number of results to return (default: 5)")
    p_search.add_argument("--section", type=str, default=None,
                          help="Filter results to a specific section")
    p_search.add_argument("--document-id", type=int, default=None,
                          help="Filter results to a specific document")
    p_search.set_defaults(func=cmd_search)

    # list
    p_list = subparsers.add_parser("list", help="List all ingested documents")
    p_list.set_defaults(func=cmd_list)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a document by ID")
    p_delete.add_argument("document_id", type=int, help="Document ID to delete")
    p_delete.set_defaults(func=cmd_delete)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
