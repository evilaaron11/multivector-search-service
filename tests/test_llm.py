"""Unit tests for core.llm — Claude CLI integration, all subprocess calls mocked."""

import sys

sys.path.insert(0, "C:/Users/aaron/Documents/embedded")

import json
from unittest.mock import patch, MagicMock

import pytest

from core.llm import (
    _call_claude,
    _call_claude_json,
    generate_node_summary,
    generate_leaf_description,
    propose_placement,
    llm_route_search,
    confirm_duplicate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_subprocess_result(stdout="", stderr="", returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# _call_claude
# ---------------------------------------------------------------------------

class TestCallClaude:

    @patch("core.llm.subprocess.run")
    def test_basic_call(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="Hello!")
        result = _call_claude("test prompt")
        assert result == "Hello!"
        mock_run.assert_called_once()

    @patch("core.llm.subprocess.run")
    def test_call_with_system(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="response")
        _call_claude("prompt", system="be helpful")

        # System instructions are combined into the input piped via stdin
        kwargs = mock_run.call_args[1]
        assert "be helpful" in kwargs["input"]
        assert "prompt" in kwargs["input"]

    @patch("core.llm.subprocess.run")
    def test_nonzero_return_raises(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="something went wrong", returncode=1
        )
        with pytest.raises(RuntimeError, match="Claude CLI error"):
            _call_claude("fail")

    @patch("core.llm.subprocess.run")
    def test_strips_whitespace(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="  trimmed  \n")
        assert _call_claude("test") == "trimmed"


# ---------------------------------------------------------------------------
# _call_claude_json
# ---------------------------------------------------------------------------

class TestCallClaudeJson:

    @patch("core.llm.subprocess.run")
    def test_parses_json(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='{"key": "value", "num": 42}'
        )
        result = _call_claude_json("test")
        assert result == {"key": "value", "num": 42}

    @patch("core.llm.subprocess.run")
    def test_strips_code_fences(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='```json\n{"key": "value"}\n```'
        )
        result = _call_claude_json("test")
        assert result == {"key": "value"}

    @patch("core.llm.subprocess.run")
    def test_fallback_on_invalid_json(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="not json at all")
        result = _call_claude_json("test")
        assert "raw" in result
        assert result["raw"] == "not json at all"

    @patch("core.llm.subprocess.run")
    def test_handles_code_fence_with_language(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='```json\n{"a": 1}\n```\n'
        )
        result = _call_claude_json("test")
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# generate_node_summary
# ---------------------------------------------------------------------------

class TestGenerateNodeSummary:

    @patch("core.llm.subprocess.run")
    def test_returns_string(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="This node contains financial news articles."
        )
        result = generate_node_summary("Finance News", ["Bloomberg articles", "WSJ reports"])
        assert isinstance(result, str)
        assert "financial" in result.lower()

    @patch("core.llm.subprocess.run")
    def test_prompt_includes_children(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="summary")
        generate_node_summary("Root", ["child1 desc", "child2 desc"])

        # Prompt is piped via stdin
        kwargs = mock_run.call_args[1]
        assert "child1 desc" in kwargs["input"]
        assert "child2 desc" in kwargs["input"]


# ---------------------------------------------------------------------------
# generate_leaf_description
# ---------------------------------------------------------------------------

class TestGenerateLeafDescription:

    @patch("core.llm.subprocess.run")
    def test_returns_string(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="A text about nuclear programs."
        )
        result = generate_leaf_description("Iran nuclear program text here...")
        assert isinstance(result, str)

    @patch("core.llm.subprocess.run")
    def test_truncates_long_text(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="summary")
        long_text = "word " * 10000
        generate_leaf_description(long_text)

        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]
        # Should be truncated to ~3000 chars of the input text
        assert len(prompt) < len(long_text)


# ---------------------------------------------------------------------------
# propose_placement
# ---------------------------------------------------------------------------

class TestProposePlacement:

    @patch("core.llm.subprocess.run")
    def test_attach_to_existing(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout=json.dumps({
                "action": "attach_to_existing",
                "target_node_id": 5,
                "new_branch_name": None,
                "new_branch_description": None,
                "confidence": 0.9,
                "reasoning": "Content matches Finance category",
            })
        )
        result = propose_placement(
            "Financial article",
            "Root: Finance (3 children)",
            [{"id": 5, "name": "Finance", "description": "Financial news", "score": 0.85}],
        )
        assert result["action"] == "attach_to_existing"
        assert result["target_node_id"] == 5
        assert result["confidence"] == 0.9

    @patch("core.llm.subprocess.run")
    def test_create_new_branch(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout=json.dumps({
                "action": "create_new_branch",
                "target_node_id": None,
                "new_branch_name": "Sports",
                "new_branch_description": "Sports news and scores",
                "confidence": 0.8,
                "reasoning": "No existing category fits",
            })
        )
        result = propose_placement("Sports article", "(empty graph)", [])
        assert result["action"] == "create_new_branch"
        assert result["new_branch_name"] == "Sports"

    @patch("core.llm.subprocess.run")
    def test_defaults_on_malformed_response(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="I'm not sure")
        result = propose_placement("test", "graph", [])
        assert "action" in result
        assert "confidence" in result
        assert result["action"] == "create_new_branch"  # default


# ---------------------------------------------------------------------------
# llm_route_search
# ---------------------------------------------------------------------------

class TestLlmRouteSearch:

    @patch("core.llm.subprocess.run")
    def test_returns_list_of_ids(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="[3, 7, 1]")
        result = llm_route_search("test query", [
            {"id": 1, "name": "A", "description": "desc", "score": 0.5},
            {"id": 3, "name": "B", "description": "desc", "score": 0.8},
            {"id": 7, "name": "C", "description": "desc", "score": 0.6},
        ])
        assert result == [3, 7, 1]

    @patch("core.llm.subprocess.run")
    def test_handles_dict_response(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='{"node_ids": [5, 2]}'
        )
        result = llm_route_search("query", [
            {"id": 2, "name": "X", "description": "", "score": 0.5},
            {"id": 5, "name": "Y", "description": "", "score": 0.6},
        ])
        assert result == [5, 2]

    @patch("core.llm.subprocess.run")
    def test_fallback_on_bad_response(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="dunno")
        candidates = [
            {"id": 1, "name": "A", "description": "", "score": 0.5},
            {"id": 2, "name": "B", "description": "", "score": 0.4},
        ]
        result = llm_route_search("query", candidates)
        # Fallback returns all candidate IDs
        assert result == [1, 2]


# ---------------------------------------------------------------------------
# confirm_duplicate
# ---------------------------------------------------------------------------

class TestConfirmDuplicate:

    @patch("core.llm.subprocess.run")
    def test_is_duplicate(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='{"is_duplicate": true, "reasoning": "Same content"}'
        )
        result = confirm_duplicate("summary A", "summary A", 0.95)
        assert result["is_duplicate"] is True

    @patch("core.llm.subprocess.run")
    def test_not_duplicate(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout='{"is_duplicate": false, "reasoning": "Different topics"}'
        )
        result = confirm_duplicate("summary A", "summary B", 0.93)
        assert result["is_duplicate"] is False

    @patch("core.llm.subprocess.run")
    def test_fallback_on_bad_response(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="idk")
        result = confirm_duplicate("a", "b", 0.9)
        assert result["is_duplicate"] is False  # safe default
