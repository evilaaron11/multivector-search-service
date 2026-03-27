"""Unit tests for core.edgar_fetcher — all EDGAR API calls are mocked."""

import sys
sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

from unittest.mock import MagicMock, patch

import pytest

from core.edgar_fetcher import fetch_tenk_sections, process_ticker


# ---------------------------------------------------------------------------
# Helpers to build mock object graphs
# ---------------------------------------------------------------------------

def _make_section_mock(text: str) -> MagicMock:
    """Return a mock section whose .text() returns *text*."""
    sec = MagicMock()
    sec.text.return_value = text
    return sec


def _build_mocks(
    sections: dict[str, str | None] | None = None,
    company_name: str = "Acme Corp",
    filing_date: str = "2025-02-15",
    company_raises: Exception | None = None,
    no_filings: bool = False,
):
    """Return ``(mock_Company_cls, mock_set_identity)`` pre-wired for a test.

    *sections* maps item names to text (or ``None`` to raise KeyError).
    """
    mock_set_identity = MagicMock()

    mock_company = MagicMock()
    mock_company.name = company_name

    mock_Company_cls = MagicMock()
    if company_raises:
        mock_Company_cls.side_effect = company_raises
    else:
        mock_Company_cls.return_value = mock_company

    if no_filings:
        mock_filings = MagicMock()
        mock_filings.latest.return_value = None
        mock_filings.__bool__ = lambda self: False
        mock_company.get_filings.return_value = mock_filings
    else:
        mock_filing = MagicMock()
        mock_filing.filing_date = filing_date

        mock_tenk = MagicMock()

        def getitem(key):
            if sections is None:
                raise KeyError(key)
            if key not in sections:
                raise KeyError(key)
            val = sections[key]
            if val is None:
                raise KeyError(key)
            return _make_section_mock(val)

        mock_tenk.__getitem__ = MagicMock(side_effect=getitem)
        mock_filing.obj.return_value = mock_tenk

        mock_filings = MagicMock()
        mock_filings.latest.return_value = mock_filing
        mock_filings.__bool__ = lambda self: True
        mock_company.get_filings.return_value = mock_filings

    return mock_Company_cls, mock_set_identity


# ---------------------------------------------------------------------------
# fetch_tenk_sections tests
# ---------------------------------------------------------------------------

class TestFetchTenkSections:

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_returns_correct_format(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={
            "Item 1": "Business description text here.",
            "Item 1A": "Risk factors text here.",
        })
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        sections, metadata = fetch_tenk_sections("ACME")

        assert isinstance(sections, list)
        assert len(sections) == 2
        for sec in sections:
            assert "section" in sec
            assert "text" in sec
        assert sections[0]["section"] == "Item 1"
        assert sections[0]["text"] == "Business description text here."

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_skips_sections_that_raise_keyerror(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={
            "Item 1": "Some text.",
            "Item 1A": None,  # will raise KeyError
        })
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        sections, _ = fetch_tenk_sections("ACME")

        names = [s["section"] for s in sections]
        assert "Item 1" in names
        assert "Item 1A" not in names

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_skips_sections_with_empty_text(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={
            "Item 1": "Has content.",
            "Item 7": "",
            "Item 8": "   ",
        })
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        sections, _ = fetch_tenk_sections("ACME")

        names = [s["section"] for s in sections]
        assert "Item 1" in names
        assert "Item 7" not in names
        assert "Item 8" not in names

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_raises_valueerror_for_company_not_found(self, mock_Company, mock_si):
        mock_Company.side_effect = Exception("Not found")

        with pytest.raises(ValueError, match="Could not find company"):
            fetch_tenk_sections("INVALID")

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_raises_valueerror_when_no_10k_filings(self, mock_Company, mock_si):
        cls, si = _build_mocks(no_filings=True)
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        with pytest.raises(ValueError, match="No 10-K filings found"):
            fetch_tenk_sections("ACME")

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_metadata_contains_company_info(self, mock_Company, mock_si):
        cls, si = _build_mocks(
            sections={"Item 1": "Text."},
            company_name="Acme Corp",
            filing_date="2025-02-15",
        )
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        _, metadata = fetch_tenk_sections("acme")

        assert metadata["ticker"] == "ACME"
        assert metadata["company_name"] == "Acme Corp"
        assert metadata["filing_date"] == "2025-02-15"
        assert metadata["form_type"] == "10-K"


# ---------------------------------------------------------------------------
# process_ticker tests
# ---------------------------------------------------------------------------

class TestProcessTicker:

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_returns_chunks_and_metadata(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={
            "Item 1": "Short business description.",
            "Item 7": "Management discussion and analysis content.",
        })
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        chunks, metadata = process_ticker("ACME")

        assert isinstance(chunks, list)
        assert isinstance(metadata, dict)
        assert len(chunks) >= 2  # at least one chunk per section

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_metadata_has_expected_keys(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={"Item 1": "Text."})
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        _, metadata = process_ticker("ACME")

        for key in ("ticker", "company_name", "filing_date", "form_type"):
            assert key in metadata, f"Missing key: {key}"

    @patch("core.edgar_fetcher.set_identity")
    @patch("core.edgar_fetcher.Company")
    def test_chunks_have_correct_keys(self, mock_Company, mock_si):
        cls, si = _build_mocks(sections={"Item 1": "Some text about the business."})
        mock_Company.side_effect = cls.side_effect
        mock_Company.return_value = cls.return_value
        mock_si.side_effect = si.side_effect

        chunks, _ = process_ticker("ACME")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "section" in chunk
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "token_count" in chunk
