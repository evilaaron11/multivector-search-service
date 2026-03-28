"""LLM integration via Claude Code CLI for routing, summarization, and dedup."""

import json
import shutil
import subprocess
import sys

from config import CLAUDE_CLI_PATH


def _resolve_cli_path() -> str:
    """Resolve the Claude CLI path, preferring .cmd on Windows."""
    resolved = shutil.which(CLAUDE_CLI_PATH)
    if resolved:
        return resolved
    # Fallback: try .cmd suffix on Windows
    if sys.platform == "win32":
        resolved = shutil.which(CLAUDE_CLI_PATH + ".cmd")
        if resolved:
            return resolved
    return CLAUDE_CLI_PATH


def _call_claude(prompt: str, system: str = "") -> str:
    """Call Claude via the CLI and return the text response.

    Uses `claude -p` (print mode) for non-interactive single-shot prompts.
    """
    cli = _resolve_cli_path()
    cmd = [cli, "-p"]

    # Combine system instructions directly into the prompt.
    # We pipe via stdin so Claude Code doesn't inject working-directory context.
    full_prompt = f"[INSTRUCTIONS: {system}]\n\n{prompt}" if system else prompt

    result = subprocess.run(
        cmd,
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr.strip()}")

    return result.stdout.strip()


def _call_claude_json(prompt: str, system: str = "") -> dict:
    """Call Claude and parse the response as JSON.

    Falls back to returning {"raw": response} if JSON parsing fails.
    """
    response = _call_claude(prompt, system)

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": response}


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You are a knowledge graph organizer. You produce concise, factual summaries. "
    "Always respond with plain text only, no markdown formatting."
)


def generate_node_summary(node_name: str, children_descriptions: list[str]) -> str:
    """Generate a 1-2 sentence summary for a branch node based on its children."""
    children_text = "\n".join(f"- {desc}" for desc in children_descriptions)
    prompt = (
        "TASK: Write a 1-2 sentence description of what information this node contains. "
        "Be specific about topics and domains covered. "
        "Respond ONLY with the description, nothing else.\n\n"
        f"Node name: {node_name}\n"
        f"Child contents:\n{children_text}"
    )
    return _call_claude(prompt, system=SUMMARY_SYSTEM)


def generate_leaf_description(text: str) -> str:
    """Generate a short description of a chunk's content."""
    # Truncate to avoid huge prompts
    truncated = text[:3000] if len(text) > 3000 else text
    prompt = (
        "TASK: Summarize the following document text in 1-2 sentences. "
        "Be specific about the key topics, entities, and facts mentioned. "
        "Respond ONLY with the summary, nothing else.\n\n"
        f"---BEGIN TEXT---\n{truncated}\n---END TEXT---"
    )
    return _call_claude(prompt, system=SUMMARY_SYSTEM)


# ---------------------------------------------------------------------------
# Ingestion routing
# ---------------------------------------------------------------------------

ROUTING_SYSTEM = (
    "You are a knowledge graph routing assistant. You decide where new content "
    "should be placed in an existing graph hierarchy. "
    "Always respond in valid JSON only, no markdown formatting or code fences."
)


def propose_placement(
    content_summary: str,
    graph_description: str,
    candidate_nodes: list[dict],
) -> dict:
    """Ask the LLM where new content should be placed in the graph.

    Args:
        content_summary: Summary of the content being ingested.
        graph_description: Text description of the current graph structure.
        candidate_nodes: List of dicts with id, name, description, score.

    Returns dict with keys:
        action: "attach_to_existing" | "create_new_branch"
        target_node_id: int | None (for attach_to_existing)
        new_branch_name: str | None (for create_new_branch)
        new_branch_description: str | None (for create_new_branch)
        confidence: float (0-1)
        reasoning: str
    """
    candidates_text = "\n".join(
        f"  - ID {c['id']}: \"{c['name']}\" (score: {c.get('score', 'N/A')}) — {c.get('description', 'no description')}"
        for c in candidate_nodes
    )
    prompt = (
        f"New content summary: {content_summary}\n\n"
        f"Current graph structure:\n{graph_description}\n\n"
        f"Candidate parent nodes:\n{candidates_text}\n\n"
        "Where should this content be placed? Respond with JSON:\n"
        '{"action": "attach_to_existing" or "create_new_branch", '
        '"target_node_id": <int or null>, '
        '"new_branch_name": "<string or null>", '
        '"new_branch_description": "<string or null>", '
        '"confidence": <0-1 float>, '
        '"reasoning": "<brief explanation>"}'
    )
    result = _call_claude_json(prompt, system=ROUTING_SYSTEM)

    # Ensure expected keys exist with defaults
    defaults = {
        "action": "create_new_branch",
        "target_node_id": None,
        "new_branch_name": "Uncategorized",
        "new_branch_description": "",
        "confidence": 0.5,
        "reasoning": result.get("raw", "LLM response could not be parsed"),
    }
    for key, default in defaults.items():
        result.setdefault(key, default)

    return result


# ---------------------------------------------------------------------------
# Search routing
# ---------------------------------------------------------------------------

def llm_route_search(
    query: str,
    candidates: list[dict],
) -> list[int]:
    """When scores are ambiguous, ask the LLM which nodes to descend into.

    Args:
        query: The search query.
        candidates: List of dicts with id, name, description, score.

    Returns ordered list of node IDs to explore.
    """
    candidates_text = "\n".join(
        f"  - ID {c['id']}: \"{c['name']}\" (score: {c['score']:.3f}) — {c.get('description', 'no description')}"
        for c in candidates
    )
    prompt = (
        f"Search query: \"{query}\"\n\n"
        f"Candidate nodes to explore:\n{candidates_text}\n\n"
        "Which nodes are most likely to contain information relevant to this query? "
        "Respond with a JSON array of node IDs in order of relevance, e.g. [3, 7, 1]. "
        "Only include nodes that are clearly relevant."
    )
    result = _call_claude_json(prompt, system=ROUTING_SYSTEM)

    # Handle both array and dict responses
    if isinstance(result, list):
        return [int(x) for x in result]
    if isinstance(result, dict) and "raw" not in result:
        # Try to find a list value in the dict
        for v in result.values():
            if isinstance(v, list):
                return [int(x) for x in v]
    return [c["id"] for c in candidates]  # fallback: return all


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def confirm_duplicate(
    new_summary: str,
    existing_summary: str,
    similarity_score: float,
) -> dict:
    """After embedding similarity flags a potential duplicate, ask LLM to confirm.

    Returns dict with keys:
        is_duplicate: bool
        reasoning: str
    """
    prompt = (
        f"New content: {new_summary}\n\n"
        f"Existing content: {existing_summary}\n\n"
        f"Embedding similarity score: {similarity_score:.3f}\n\n"
        "Are these describing the same content (a duplicate)? "
        'Respond with JSON: {"is_duplicate": true/false, "reasoning": "..."}'
    )
    result = _call_claude_json(prompt, system=ROUTING_SYSTEM)
    return {
        "is_duplicate": result.get("is_duplicate", False),
        "reasoning": result.get("reasoning", result.get("raw", "")),
    }
