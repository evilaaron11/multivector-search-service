"""RSS/Atom feed parser and web article fetcher.

Fetches articles from RSS feeds, extracts content with trafilatura,
and chunks them using the existing chunker pipeline.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

from core.chunker import parse_sections, chunk_section, estimate_tokens
from config import (
    RSS_FETCH_DELAY,
    RSS_REQUEST_TIMEOUT,
    RSS_MAX_ARTICLES_PER_FEED,
    RSS_MIN_ARTICLE_WORDS,
)

USER_AGENT = "EmbeddingPipeline/1.0 (RSS reader)"

# Atom namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"


# ---------------------------------------------------------------------------
# Feed name validation
# ---------------------------------------------------------------------------

def validate_feed_name(name: str | None) -> str | None:
    """Validate a user-provided feed name.

    Returns the cleaned name if valid, or None if it should be auto-detected.
    """
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    if name.startswith("http://") or name.startswith("https://"):
        return None
    if len(name) > 100:
        return None
    return name


def _extract_domain(url: str) -> str:
    """Extract a readable name from a URL's domain."""
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    # Strip www. prefix
    if domain.startswith("www."):
        domain = domain[4:]
    # Take the first part before the TLD
    parts = domain.split(".")
    if parts:
        return parts[0]
    return domain


def resolve_feed_name(
    user_name: str | None,
    feed_title: str | None,
    feed_url: str,
) -> str:
    """Resolve a feed name from user input, feed metadata, or URL.

    Priority: validated user name -> feed channel title -> domain from URL.
    """
    validated = validate_feed_name(user_name)
    if validated:
        return validated

    if feed_title and feed_title.strip():
        return feed_title.strip()[:100]

    return _extract_domain(feed_url)


# ---------------------------------------------------------------------------
# Feeds file loading
# ---------------------------------------------------------------------------

def load_feeds_file(file_path: str) -> list[dict]:
    """Load a feeds JSON file.

    Expected format: [{"url": "...", "name": "..."}, ...]
    The "name" field is optional.

    Returns list of dicts with at least "url" key.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Feeds file not found: {file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Feeds file must contain a JSON array")

    feeds = []
    for entry in data:
        if isinstance(entry, str):
            feeds.append({"url": entry, "name": None})
        elif isinstance(entry, dict) and "url" in entry:
            feeds.append(entry)
        else:
            raise ValueError(f"Invalid feed entry: {entry}")

    return feeds


# ---------------------------------------------------------------------------
# RSS / Atom parsing
# ---------------------------------------------------------------------------

def parse_feed(feed_xml: str) -> tuple[str | None, list[dict]]:
    """Parse RSS or Atom XML into article entries.

    Returns:
        (feed_title, articles) where articles is a list of dicts:
        [{"title": str, "url": str, "published": str, "summary": str}]
    """
    root = ET.fromstring(feed_xml)

    # Detect format
    if root.tag == "rss":
        return _parse_rss(root)
    elif root.tag == f"{ATOM_NS}feed" or root.tag == "feed":
        return _parse_atom(root)
    else:
        raise ValueError(f"Unknown feed format: root tag is <{root.tag}>")


def _parse_rss(root: ET.Element) -> tuple[str | None, list[dict]]:
    channel = root.find("channel")
    if channel is None:
        return None, []

    feed_title = _text(channel, "title")
    articles = []

    for item in channel.findall("item"):
        title = _text(item, "title") or "Untitled"
        link = _text(item, "link") or ""
        pub_date = _text(item, "pubDate") or ""
        summary = _text(item, "description") or ""

        if link:
            articles.append({
                "title": title,
                "url": link.strip(),
                "published": pub_date,
                "summary": summary,
            })

    return feed_title, articles


def _parse_atom(root: ET.Element) -> tuple[str | None, list[dict]]:
    ns = ATOM_NS if root.tag.startswith("{") else ""

    feed_title = _text(root, f"{ns}title")
    articles = []

    for entry in root.findall(f"{ns}entry"):
        title = _text(entry, f"{ns}title") or "Untitled"

        # Atom links can be in <link href="..."> elements
        link_el = entry.find(f"{ns}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{ns}link")
        url = link_el.get("href", "") if link_el is not None else ""

        published = _text(entry, f"{ns}published") or _text(entry, f"{ns}updated") or ""
        summary = _text(entry, f"{ns}summary") or _text(entry, f"{ns}content") or ""

        if url:
            articles.append({
                "title": title,
                "url": url.strip(),
                "published": published,
                "summary": summary,
            })

    return feed_title, articles


def _text(parent: ET.Element, tag: str) -> str | None:
    el = parent.find(tag)
    if el is not None and el.text:
        return el.text.strip()
    return None


# ---------------------------------------------------------------------------
# Article fetching and content extraction
# ---------------------------------------------------------------------------

def fetch_article(url: str) -> tuple[str, dict]:
    """Fetch a URL and extract article text with trafilatura.

    Returns (cleaned_text, metadata).
    Raises ValueError if content cannot be fetched or is too short.
    """
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=RSS_REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    html = response.text
    text = trafilatura.extract(
        html, output_format="markdown", include_links=False, include_tables=False,
    )

    if not text:
        raise ValueError(f"Could not extract content from {url}")

    word_count = len(text.split())
    if word_count < RSS_MIN_ARTICLE_WORDS:
        raise ValueError(
            f"Article too short ({word_count} words, min {RSS_MIN_ARTICLE_WORDS}): {url}"
        )

    # Try to get the title from trafilatura metadata, then fall back to <title> tag
    title = None
    try:
        doc_meta = trafilatura.bare_extraction(html)
        if doc_meta and getattr(doc_meta, "title", None):
            title = doc_meta.title
    except Exception:
        pass

    if not title:
        # Fall back to HTML <title> tag
        import re
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            title = match.group(1).strip()

    return text, {
        "url": url,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def process_url(url: str, source_name: str | None = None) -> tuple[list[dict], dict]:
    """Fetch a URL, extract content, chunk it.

    Returns (chunks, metadata) in the same format as edgar_fetcher.process_ticker().
    """
    text, meta = fetch_article(url)

    # Use title from metadata, or derive from URL
    title = meta.get("title") or _extract_domain(url)

    # Chunk the text using the existing chunker
    sections = parse_sections(text)

    # Replace "Preamble" section names with the article title
    for sec in sections:
        if sec["section"] == "Preamble":
            sec["section"] = title

    all_chunks: list[dict] = []
    for sec in sections:
        all_chunks.extend(chunk_section(sec["section"], sec["text"]))

    # If no sections were found (no headers), chunk the raw text with article title
    if not all_chunks and text.strip():
        all_chunks = chunk_section(title, text.strip())

    metadata = {
        "url": url,
        "title": title,
        "source_name": source_name,
    }

    return all_chunks, metadata


def process_feed(
    feed_url: str,
    feed_name: str | None = None,
    max_articles: int | None = None,
) -> list[tuple[list[dict], dict]]:
    """Fetch an RSS feed, then fetch and chunk each article.

    Args:
        feed_url: URL of the RSS/Atom feed.
        feed_name: User-provided name for this feed (validated).
        max_articles: Max articles to process (default: config value).

    Returns list of (chunks, metadata) tuples, one per successfully fetched article.
    """
    if max_articles is None:
        max_articles = RSS_MAX_ARTICLES_PER_FEED

    # Fetch the feed XML
    response = requests.get(
        feed_url,
        headers={"User-Agent": USER_AGENT},
        timeout=RSS_REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    feed_title, articles = parse_feed(response.text)
    resolved_name = resolve_feed_name(feed_name, feed_title, feed_url)

    articles = articles[:max_articles]
    results: list[tuple[list[dict], dict]] = []

    for i, article in enumerate(articles):
        try:
            chunks, meta = process_url(article["url"], source_name=resolved_name)
            # Override title with RSS title if we got one
            if article.get("title") and article["title"] != "Untitled":
                meta["title"] = article["title"]
            meta["published"] = article.get("published", "")
            meta["source_name"] = resolved_name
            results.append((chunks, meta))
        except (ValueError, requests.RequestException) as e:
            print(f"  Skipping [{i+1}/{len(articles)}]: {e}")
            continue

        # Rate limiting between fetches
        if i < len(articles) - 1:
            time.sleep(RSS_FETCH_DELAY)

    return results
