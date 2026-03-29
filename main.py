"""CLI entry point for the DAG-based embedding pipeline.

Subcommands: ingest, ingest-ticker, search, list, delete, graph, enrich, migrate.
"""

from __future__ import annotations

import argparse
import sys

# Ensure Unicode output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import service
from core import store
from core.graph import get_graph_description


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _print_ingest(result: service.IngestResult) -> None:
    if result.node_id == -1:
        print(f"Skipped (duplicate): {result.name}")
        return
    print(f"Ingested: {result.name}")
    print(f"  Node ID     : {result.node_id}")
    print(f"  Chunks      : {result.num_chunks}")
    print(f"  Sections    : {', '.join(result.sections)}")
    if result.parent_node_id is not None:
        parent = store.get_node(result.parent_node_id)
        parent_name = parent["name"] if parent else str(result.parent_node_id)
        print(f"  Placed under: {parent_name}")
    if result.is_new_branch:
        print(f"  (new category created)")


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "were", "be",
    "been", "has", "had", "have", "that", "this", "not", "are", "as", "—",
    "he", "she", "we", "they", "if", "so", "no", "do", "did", "than",
})


def _is_content_word(word: str) -> bool:
    clean = word.strip(".,;:!?\"'()-\u2013\u2014*[]{}|/").lower()
    return len(clean) > 2 and clean not in _STOPWORDS


def _highlight_text(highlights: list[tuple[str, float, str]], threshold: float = 0.72) -> str:
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


def _print_search(results: list[service.SearchResult]) -> None:
    if not results:
        print("No results found.")
        return

    sep = "\u2501" * 60

    for rank, r in enumerate(results, start=1):
        print(sep)
        print(f"[{rank}] Score: {r.score:.4f}")
        print(f"    Name: {r.name}")
        print(f"    Type: {r.node_type}")
        print()

        if r.highlights:
            highlighted = _highlight_text(r.highlights)
            words = highlighted.split(" ")
            lines = []
            current_line = "    "
            for w in words:
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
    print(f"\n\033[1;31m\u25a0\033[0m strong match  \033[1;33m\u25a0\033[0m good match")


def _print_list_tree(nodes: list[service.NodeInfo]) -> None:
    """Print nodes as a condensed tree with box-drawing characters."""
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    TYPE_COLORS = {"category": CYAN, "document": YELLOW, "chunk": GREEN}

    if not nodes:
        print("No nodes in the graph.")
        return

    roots = store.get_roots()
    if not roots:
        print("No nodes in the graph.")
        return

    def _walk(node_id: int, prefix: str = "", is_last: bool = True, depth: int = 0) -> None:
        node = store.get_node(node_id)
        if not node:
            return

        children = store.get_children(node_id)
        ntype = node["node_type"]
        color = TYPE_COLORS.get(ntype, "")

        # Connector characters
        if depth == 0:
            connector = ""
            child_prefix = ""
        else:
            connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
            child_prefix = prefix + ("   " if is_last else "\u2502  ")

        # Build the display line
        type_tag = f"{color}{ntype}{RESET}"
        name = node["name"]

        if ntype == "chunk":
            # For chunks, show condensed: just count them at parent level
            return  # handled by parent
        elif ntype == "document":
            chunk_count = len(children)
            line = f"{prefix}{connector}{type_tag} {BOLD}{name}{RESET} {DIM}({chunk_count} chunks){RESET}"
            print(line)
        else:  # category
            doc_children = [c for c in children if c["node_type"] != "chunk"]
            desc = node.get("description", "")
            if desc.startswith("Source:"):
                desc = desc.split("\n\n", 1)[1] if "\n\n" in desc else ""
            desc_preview = f" {DIM}— {desc[:70]}...{RESET}" if len(desc) > 70 else (f" {DIM}— {desc}{RESET}" if desc else "")
            line = f"{prefix}{connector}{type_tag} {BOLD}{name}{RESET}{desc_preview}"
            print(line)

            for i, child in enumerate(doc_children):
                _walk(child["id"], child_prefix, is_last=(i == len(doc_children) - 1), depth=depth + 1)

    for i, root in enumerate(roots):
        if i > 0:
            print()
        _walk(root["id"], depth=0)



def _print_list_flat(nodes: list[service.NodeInfo]) -> None:
    """Print nodes as a flat table."""
    if not nodes:
        print("No nodes in the graph.")
        return

    id_w = max(len("ID"), max(len(str(n.node_id)) for n in nodes))
    tp_w = max(len("Type"), max(len(n.node_type) for n in nodes))
    nm_w = min(40, max(len("Name"), max(len(n.name) for n in nodes)))

    header = f"{'ID':>{id_w}}  {'Type':<{tp_w}}  {'Name':<{nm_w}}  Children"
    print(header)
    print("-" * len(header))
    for n in nodes:
        name_display = n.name[:nm_w] if len(n.name) > nm_w else n.name
        print(f"{n.node_id:>{id_w}}  {n.node_type:<{tp_w}}  {name_display:<{nm_w}}  {n.child_count}")


# ---------------------------------------------------------------------------
# Interactive confirmation for ingestion
# ---------------------------------------------------------------------------

def _cli_confirm_callback(question: str, options: list[dict]) -> int:
    """Prompt the user to pick an option during ingestion."""
    print(f"\n? {question}\n")
    for i, opt in enumerate(options):
        label = opt.get("label", str(opt))
        print(f"  [{i + 1}] {label}")
    print()

    while True:
        try:
            raw = input(f"Enter choice [1]: ").strip()
            if not raw:
                return 0
            choice = int(raw) - 1
            if 0 <= choice < len(options):
                return choice
            print(f"  Please enter 1-{len(options)}")
        except (ValueError, EOFError):
            return 0


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    callback = None if args.autonomous else _cli_confirm_callback
    result = service.ingest_document(args.file_path, confirm_callback=callback, force=args.force)
    _print_ingest(result)


def cmd_ingest_ticker(args: argparse.Namespace) -> None:
    callback = None if args.autonomous else _cli_confirm_callback
    result = service.ingest_ticker(args.ticker, confirm_callback=callback, force=args.force)
    _print_ingest(result)


def _print_trace(trace: list[dict]) -> None:
    """Print graph traversal trace."""
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Graph traversal:{RESET}")
    for step in trace:
        depth = step["depth"]
        action = step["action"]
        indent = "  " * depth

        if action == "fallback_to_flat":
            print(f"{indent}{DIM}(no embeddings on roots, falling back to flat search){RESET}")
            continue

        label = f"Level {depth}" if action == "score_and_select" else f"Level {depth} (leaves)"
        print(f"\n{indent}{BOLD}{label}{RESET}")

        for node in step["nodes"]:
            score_str = f"{node['score']:.4f}"
            name = node["name"]
            ntype = node.get("type", "")

            if node.get("leaf"):
                marker = f"{CYAN}\u2514\u2500 leaf{RESET}"
            elif node.get("selected"):
                marker = f"{GREEN}\u2714 descend{RESET}"
            else:
                marker = f"{RED}\u2718 skip{RESET}"

            print(f"{indent}  {marker}  {score_str}  [{ntype}] {name}")

    print()


def cmd_search(args: argparse.Namespace) -> None:
    verbose = getattr(args, "verbose", False) and not args.flat

    if verbose:
        results, trace = service.search_documents(
            query=args.query,
            top_k=args.top_k,
            flat=args.flat,
            verbose=True,
        )
        _print_trace(trace)
    else:
        results = service.search_documents(
            query=args.query,
            top_k=args.top_k,
            flat=args.flat,
        )

    _print_search(results)


def cmd_list(args: argparse.Namespace) -> None:
    nodes = service.list_nodes()
    if args.flat:
        _print_list_flat(nodes)
    else:
        _print_list_tree(nodes)


def cmd_delete(args: argparse.Namespace) -> None:
    deleted = service.delete_node(args.node_id)
    if deleted:
        print(f"Node {args.node_id} and descendants deleted.")
    else:
        print(f"Node {args.node_id} not found.")


def cmd_graph(args: argparse.Namespace) -> None:
    store.init_db()
    desc = get_graph_description(max_depth=args.depth)
    print(desc)


def cmd_enrich(args: argparse.Namespace) -> None:
    count = service.enrich_graph(node_id=args.node)
    print(f"Enriched {count} nodes with summaries and embeddings.")


def cmd_ingest_url(args: argparse.Namespace) -> None:
    callback = None if args.autonomous else _cli_confirm_callback
    result = service.ingest_url(args.url, confirm_callback=callback, force=args.force)
    _print_ingest(result)


def cmd_ingest_feed(args: argparse.Namespace) -> None:
    results = service.ingest_feed(
        args.feed_url,
        feed_name=getattr(args, "name", None),
        max_articles=args.max_articles,
        confirm_callback=None,
        force=args.force,
    )
    ingested = [r for r in results if r.node_id != -1]
    skipped = len(results) - len(ingested)
    print(f"\nDone. Ingested {len(ingested)} articles ({skipped} skipped).")


def cmd_ingest_feeds(args: argparse.Namespace) -> None:
    results = service.ingest_all_feeds(
        args.feeds_file,
        max_articles=args.max_articles,
        confirm_callback=None,
        force=args.force,
    )
    ingested = [r for r in results if r.node_id != -1]
    skipped = len(results) - len(ingested)
    print(f"\nDone. Ingested {len(ingested)} articles total ({skipped} skipped).")


def cmd_migrate(args: argparse.Namespace) -> None:
    store.init_db()
    print("Migration check complete. Database is up to date.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DAG-based embedding pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a markdown document")
    p_ingest.add_argument("file_path", help="Path to the markdown file")
    p_ingest.add_argument("--autonomous", action="store_true",
                          help="Skip interactive confirmations")
    p_ingest.add_argument("--force", action="store_true",
                          help="Delete existing and re-ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    # ingest-ticker
    p_ticker = subparsers.add_parser(
        "ingest-ticker", help="Ingest an SEC filing by stock ticker",
    )
    p_ticker.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    p_ticker.add_argument("--autonomous", action="store_true",
                          help="Skip interactive confirmations")
    p_ticker.add_argument("--force", action="store_true",
                          help="Delete existing and re-ingest")
    p_ticker.set_defaults(func=cmd_ingest_ticker)

    # search
    p_search = subparsers.add_parser("search", help="Search the knowledge graph")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=5,
                          help="Number of results (default: 5)")
    p_search.add_argument("--flat", action="store_true",
                          help="Use flat search (no graph navigation)")
    p_search.add_argument("--verbose", "-v", action="store_true",
                          help="Show graph traversal trace")
    p_search.set_defaults(func=cmd_search)

    # list
    p_list = subparsers.add_parser("list", help="List nodes in the graph")
    p_list.add_argument("--flat", action="store_true",
                        help="Show flat table instead of tree view")
    p_list.set_defaults(func=cmd_list)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a node and descendants")
    p_delete.add_argument("node_id", type=int, help="Node ID to delete")
    p_delete.set_defaults(func=cmd_delete)

    # graph
    p_graph = subparsers.add_parser("graph", help="Show graph structure")
    p_graph.add_argument("--depth", type=int, default=3,
                         help="Max depth to display (default: 3)")
    p_graph.set_defaults(func=cmd_graph)

    # enrich
    p_enrich = subparsers.add_parser("enrich",
                                     help="Generate summaries/embeddings for nodes missing them")
    p_enrich.add_argument("--node", type=int, default=None,
                          help="Enrich only a specific subtree")
    p_enrich.set_defaults(func=cmd_enrich)

    # ingest-url
    p_url = subparsers.add_parser("ingest-url", help="Ingest a web page by URL")
    p_url.add_argument("url", help="URL of the web page to ingest")
    p_url.add_argument("--autonomous", action="store_true",
                        help="Skip interactive confirmations")
    p_url.add_argument("--force", action="store_true",
                        help="Delete existing and re-ingest")
    p_url.set_defaults(func=cmd_ingest_url)

    # ingest-feed
    p_feed = subparsers.add_parser("ingest-feed", help="Ingest articles from an RSS feed")
    p_feed.add_argument("feed_url", help="URL of the RSS/Atom feed")
    p_feed.add_argument("--name", default=None,
                         help="Display name for the feed source")
    p_feed.add_argument("--max-articles", type=int, default=None,
                         help="Max articles to ingest (default: config value)")
    p_feed.add_argument("--force", action="store_true",
                         help="Delete existing and re-ingest all articles")
    p_feed.set_defaults(func=cmd_ingest_feed)

    # ingest-feeds
    p_feeds = subparsers.add_parser("ingest-feeds",
                                     help="Ingest all feeds from a JSON file")
    p_feeds.add_argument("feeds_file", help="Path to feeds JSON file")
    p_feeds.add_argument("--max-articles", type=int, default=None,
                          help="Max articles per feed (default: config value)")
    p_feeds.add_argument("--force", action="store_true",
                          help="Delete existing and re-ingest all articles")
    p_feeds.set_defaults(func=cmd_ingest_feeds)

    # migrate
    p_migrate = subparsers.add_parser("migrate",
                                      help="Run database migration from v1 to v2")
    p_migrate.set_defaults(func=cmd_migrate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
