"""Unit tests for core.web_fetcher — RSS parsing, HTML extraction, feed name validation."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import patch, MagicMock

import pytest

from core.web_fetcher import (
    validate_feed_name,
    resolve_feed_name,
    parse_feed,
    load_feeds_file,
    process_url,
    _extract_domain,
)


# ---------------------------------------------------------------------------
# Feed name validation
# ---------------------------------------------------------------------------

class TestValidateFeedName:

    def test_valid_name(self):
        assert validate_feed_name("TechCrunch") == "TechCrunch"

    def test_strips_whitespace(self):
        assert validate_feed_name("  Ars Technica  ") == "Ars Technica"

    def test_rejects_empty(self):
        assert validate_feed_name("") is None
        assert validate_feed_name("   ") is None
        assert validate_feed_name(None) is None

    def test_rejects_url(self):
        assert validate_feed_name("https://techcrunch.com") is None
        assert validate_feed_name("http://example.com") is None

    def test_rejects_too_long(self):
        assert validate_feed_name("x" * 101) is None

    def test_accepts_max_length(self):
        assert validate_feed_name("x" * 100) == "x" * 100


class TestResolveFeedName:

    def test_prefers_user_name(self):
        assert resolve_feed_name("MyFeed", "Channel Title", "https://example.com/feed") == "MyFeed"

    def test_falls_back_to_feed_title(self):
        assert resolve_feed_name(None, "Channel Title", "https://example.com/feed") == "Channel Title"

    def test_falls_back_to_domain(self):
        assert resolve_feed_name(None, None, "https://techcrunch.com/feed/") == "techcrunch"

    def test_rejects_invalid_user_name_falls_to_title(self):
        assert resolve_feed_name("https://bad.com", "Good Title", "https://x.com") == "Good Title"

    def test_rejects_invalid_user_name_falls_to_domain(self):
        assert resolve_feed_name("https://bad.com", None, "https://example.com") == "example"


class TestExtractDomain:

    def test_simple_domain(self):
        assert _extract_domain("https://techcrunch.com/feed/") == "techcrunch"

    def test_www_stripped(self):
        assert _extract_domain("https://www.theverge.com/rss") == "theverge"

    def test_subdomain(self):
        assert _extract_domain("https://blog.google/rss") == "blog"


# ---------------------------------------------------------------------------
# RSS / Atom parsing
# ---------------------------------------------------------------------------

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/article-1</link>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
      <description>First article summary</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/article-2</link>
      <pubDate>Tue, 02 Jan 2024 12:00:00 GMT</pubDate>
      <description>Second article summary</description>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <entry>
    <title>Atom Article</title>
    <link rel="alternate" href="https://example.com/atom-1"/>
    <published>2024-01-01T12:00:00Z</published>
    <summary>An atom article summary</summary>
  </entry>
</feed>
"""


class TestParseFeedRSS:

    def test_parses_rss_title(self):
        title, articles = parse_feed(SAMPLE_RSS)
        assert title == "Test Feed"

    def test_parses_rss_articles(self):
        _, articles = parse_feed(SAMPLE_RSS)
        assert len(articles) == 2

    def test_rss_article_fields(self):
        _, articles = parse_feed(SAMPLE_RSS)
        assert articles[0]["title"] == "Article One"
        assert articles[0]["url"] == "https://example.com/article-1"
        assert "Mon" in articles[0]["published"]
        assert articles[0]["summary"] == "First article summary"

    def test_rss_second_article(self):
        _, articles = parse_feed(SAMPLE_RSS)
        assert articles[1]["title"] == "Article Two"
        assert articles[1]["url"] == "https://example.com/article-2"


class TestParseFeedAtom:

    def test_parses_atom_title(self):
        title, articles = parse_feed(SAMPLE_ATOM)
        assert title == "Atom Test Feed"

    def test_parses_atom_articles(self):
        _, articles = parse_feed(SAMPLE_ATOM)
        assert len(articles) == 1

    def test_atom_article_fields(self):
        _, articles = parse_feed(SAMPLE_ATOM)
        assert articles[0]["title"] == "Atom Article"
        assert articles[0]["url"] == "https://example.com/atom-1"
        assert "2024" in articles[0]["published"]


class TestParseFeedEdgeCases:

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown feed format"):
            parse_feed("<html><body>not a feed</body></html>")

    def test_empty_rss_channel(self):
        xml = '<rss version="2.0"><channel><title>Empty</title></channel></rss>'
        title, articles = parse_feed(xml)
        assert title == "Empty"
        assert articles == []

    def test_item_without_link_skipped(self):
        xml = """<rss version="2.0"><channel><title>T</title>
        <item><title>No Link</title></item>
        <item><title>Has Link</title><link>https://x.com/a</link></item>
        </channel></rss>"""
        _, articles = parse_feed(xml)
        assert len(articles) == 1
        assert articles[0]["title"] == "Has Link"


# ---------------------------------------------------------------------------
# Feeds file loading
# ---------------------------------------------------------------------------

class TestLoadFeedsFile:

    def test_loads_json_array(self, tmp_path):
        f = tmp_path / "feeds.json"
        f.write_text('[{"url": "https://example.com/feed", "name": "Test"}]')
        feeds = load_feeds_file(str(f))
        assert len(feeds) == 1
        assert feeds[0]["url"] == "https://example.com/feed"
        assert feeds[0]["name"] == "Test"

    def test_loads_string_urls(self, tmp_path):
        f = tmp_path / "feeds.json"
        f.write_text('["https://example.com/feed1", "https://example.com/feed2"]')
        feeds = load_feeds_file(str(f))
        assert len(feeds) == 2
        assert feeds[0]["url"] == "https://example.com/feed1"
        assert feeds[0]["name"] is None

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_feeds_file("/nonexistent/feeds.json")

    def test_non_array_raises(self, tmp_path):
        f = tmp_path / "feeds.json"
        f.write_text('{"url": "bad"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_feeds_file(str(f))

    def test_invalid_entry_raises(self, tmp_path):
        f = tmp_path / "feeds.json"
        f.write_text('[42]')
        with pytest.raises(ValueError, match="Invalid feed entry"):
            load_feeds_file(str(f))


# ---------------------------------------------------------------------------
# process_url (mocked HTTP)
# ---------------------------------------------------------------------------

class TestProcessUrl:

    @patch("core.web_fetcher.trafilatura.bare_extraction")
    @patch("core.web_fetcher.trafilatura.extract")
    @patch("core.web_fetcher.requests.get")
    def test_returns_chunks_and_metadata(self, mock_get, mock_extract, mock_bare):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Hello world</p></body></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        long_text = "Hello world. " * 50  # well over 100 words
        mock_extract.return_value = long_text
        mock_bare.return_value = MagicMock(title="Test Article")

        chunks, metadata = process_url("https://example.com/article")

        assert len(chunks) >= 1
        assert chunks[0]["section"] is not None
        assert chunks[0]["text"] is not None
        assert chunks[0]["chunk_index"] == 0
        assert "token_count" in chunks[0]
        assert metadata["url"] == "https://example.com/article"
        assert metadata["title"] == "Test Article"

    @patch("core.web_fetcher.trafilatura.extract")
    @patch("core.web_fetcher.requests.get")
    def test_raises_on_empty_content(self, mock_get, mock_extract):
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        mock_extract.return_value = None

        with pytest.raises(ValueError, match="Could not extract"):
            process_url("https://example.com/empty")

    @patch("core.web_fetcher.trafilatura.extract")
    @patch("core.web_fetcher.requests.get")
    def test_raises_on_short_content(self, mock_get, mock_extract):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Short</body></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        mock_extract.return_value = "Just a few words"

        with pytest.raises(ValueError, match="too short"):
            process_url("https://example.com/short")

    @patch("core.web_fetcher.trafilatura.bare_extraction")
    @patch("core.web_fetcher.trafilatura.extract")
    @patch("core.web_fetcher.requests.get")
    def test_source_name_passed_through(self, mock_get, mock_extract, mock_bare):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>content</body></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        mock_extract.return_value = "Long article. " * 50
        mock_bare.return_value = {"title": "Title"}

        _, metadata = process_url("https://example.com/a", source_name="MySource")
        assert metadata["source_name"] == "MySource"


# ---------------------------------------------------------------------------
# URL dedup in store
# ---------------------------------------------------------------------------

class TestFindNodeBySourceUrl:

    @pytest.fixture(autouse=True)
    def _patch_config(self, tmp_path, monkeypatch):
        import config
        import importlib
        import core.store as store_mod

        db_path = tmp_path / "test.db"
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()

        monkeypatch.setattr(config, "DB_PATH", db_path)
        monkeypatch.setattr(config, "VECTORS_DIR", vectors_dir)
        importlib.reload(store_mod)
        monkeypatch.setattr(store_mod, "DB_PATH", db_path)
        monkeypatch.setattr(store_mod, "VECTORS_DIR", vectors_dir)

    def test_finds_by_url(self):
        from core.store import init_db, insert_node, find_node_by_source_url
        init_db()
        insert_node("document", "Test", description="Source: https://example.com/article\n\nSummary here")
        result = find_node_by_source_url("https://example.com/article")
        assert result is not None
        assert result["name"] == "Test"

    def test_returns_none_when_not_found(self):
        from core.store import init_db, find_node_by_source_url
        init_db()
        result = find_node_by_source_url("https://nonexistent.com")
        assert result is None

    def test_ignores_non_document_nodes(self):
        from core.store import init_db, insert_node, find_node_by_source_url
        init_db()
        insert_node("chunk", "Chunk", description="Source: https://example.com/x\n\ntext")
        result = find_node_by_source_url("https://example.com/x")
        assert result is None
