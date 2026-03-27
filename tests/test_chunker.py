"""
Unit tests for core.chunker.
"""

import sys
import textwrap

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

import pytest

from core.chunker import (
    estimate_tokens,
    parse_sections,
    chunk_section,
    process_markdown,
    API_TOKEN_LIMIT,
)


# ===================================================================
# estimate_tokens
# ===================================================================

class TestEstimateTokens:
    def test_simple_sentence(self):
        assert estimate_tokens("hello world foo bar") == 4

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        assert estimate_tokens("word") == 1

    def test_extra_whitespace(self):
        # Multiple spaces / tabs collapse via split()
        assert estimate_tokens("  a   b  c  ") == 3


# ===================================================================
# parse_sections
# ===================================================================

class TestParseSections:
    def test_single_section(self):
        md = "# Introduction\nSome intro text here."
        result = parse_sections(md)
        assert len(result) == 1
        assert result[0]["section"] == "Introduction"
        assert "Some intro text" in result[0]["text"]

    def test_multiple_sections(self):
        md = textwrap.dedent("""\
            # Section A
            Content A.

            # Section B
            Content B.
        """)
        result = parse_sections(md)
        assert len(result) == 2
        assert result[0]["section"] == "Section A"
        assert result[1]["section"] == "Section B"

    def test_preamble_before_first_header(self):
        md = textwrap.dedent("""\
            This is preamble text.

            # First Header
            Body of first header.
        """)
        result = parse_sections(md)
        assert result[0]["section"] == "Preamble"
        assert "preamble text" in result[0]["text"]
        assert result[1]["section"] == "First Header"

    def test_nested_h2_headers(self):
        md = textwrap.dedent("""\
            # Top Level
            Top level content.

            ## Sub Level
            Sub level content.
        """)
        result = parse_sections(md)
        assert len(result) == 2
        assert result[0]["section"] == "Top Level"
        assert result[1]["section"] == "Sub Level"

    def test_consecutive_headers_collapse(self):
        """Consecutive headers with no content should keep only the last."""
        md = textwrap.dedent("""\
            # First
            # Second
            Actual content here.
        """)
        result = parse_sections(md)
        assert len(result) == 1
        assert result[0]["section"] == "Second"
        assert "Actual content" in result[0]["text"]

    def test_empty_sections_skipped(self):
        md = textwrap.dedent("""\
            # Has Content
            Real content.

            # Empty Section
            # Also Content
            More content.
        """)
        result = parse_sections(md)
        sections = [s["section"] for s in result]
        assert "Empty Section" not in sections
        assert "Has Content" in sections
        assert "Also Content" in sections

    def test_no_headers_at_all(self):
        md = "Just plain text, no markdown headers."
        result = parse_sections(md)
        assert len(result) == 1
        assert result[0]["section"] == "Preamble"

    def test_empty_string(self):
        result = parse_sections("")
        assert result == []

    def test_only_headers_no_body(self):
        md = "# A\n# B\n# C\n"
        result = parse_sections(md)
        assert isinstance(result, list)


# ===================================================================
# chunk_section
# ===================================================================

class TestChunkSection:
    def test_short_text_single_chunk(self):
        text = "This is a short section."
        result = chunk_section("Intro", text)
        assert len(result) == 1
        assert result[0]["section"] == "Intro"
        assert result[0]["chunk_index"] == 0
        assert result[0]["token_count"] == estimate_tokens(text)

    def test_empty_text(self):
        result = chunk_section("Empty", "")
        assert result == []

    def test_whitespace_only(self):
        result = chunk_section("Blank", "   \n\n  ")
        assert result == []

    def test_hard_split_at_token_limit(self, monkeypatch):
        """Text exceeding the API token limit should be hard-split."""
        import core.chunker as chunker_mod
        monkeypatch.setattr(chunker_mod, "API_TOKEN_LIMIT", 50)

        words = [f"w{i}" for i in range(120)]
        text = " ".join(words)

        result = chunk_section("Big", text)
        assert len(result) == 3  # 50 + 50 + 20
        for chunk in result:
            assert chunk["section"] == "Big"
            assert chunk["token_count"] <= 50

        indices = [c["chunk_index"] for c in result]
        assert indices == [0, 1, 2]

    def test_text_exactly_at_limit(self, monkeypatch):
        import core.chunker as chunker_mod
        monkeypatch.setattr(chunker_mod, "API_TOKEN_LIMIT", 10)

        text = " ".join(f"w{i}" for i in range(10))
        result = chunk_section("Exact", text)
        assert len(result) == 1
        assert result[0]["token_count"] == 10


# ===================================================================
# process_markdown — end-to-end
# ===================================================================

class TestProcessMarkdown:
    def test_end_to_end(self, tmp_path):
        md_content = textwrap.dedent("""\
            # Overview
            This is the overview section with enough words to form a chunk.

            # Details
            Here are some details about the filing that span a few words.
        """)
        md_file = tmp_path / "test_filing.md"
        md_file.write_text(md_content, encoding="utf-8")

        result = process_markdown(str(md_file))
        assert len(result) >= 2
        sections_found = {c["section"] for c in result}
        assert "Overview" in sections_found
        assert "Details" in sections_found

    def test_empty_file(self, tmp_path):
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")

        result = process_markdown(str(md_file))
        assert result == []

    def test_file_with_only_headers(self, tmp_path):
        md_file = tmp_path / "headers_only.md"
        md_file.write_text("# A\n# B\n# C\n", encoding="utf-8")

        result = process_markdown(str(md_file))
        assert isinstance(result, list)

    def test_file_with_no_headers(self, tmp_path):
        md_file = tmp_path / "no_headers.md"
        md_file.write_text("Just plain text without any markdown headers.", encoding="utf-8")

        result = process_markdown(str(md_file))
        assert len(result) == 1
        assert result[0]["section"] == "Preamble"
