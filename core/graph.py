"""Graph traversal utilities for the DAG-based node structure."""

from core.store import get_roots, get_children, get_node, count_children


def get_graph_description(max_depth: int = 3) -> str:
    """Generate a text description of the graph structure for LLM prompts.

    Returns a tree-like string showing node names and child counts.
    """
    lines: list[str] = []
    roots = get_roots()

    if not roots:
        return "(empty graph)"

    def _walk(node: dict, depth: int, indent: str) -> None:
        child_count = count_children(node["id"])
        desc = node.get("description", "")
        desc_preview = f" — {desc[:80]}..." if len(desc) > 80 else (f" — {desc}" if desc else "")

        if child_count > 0:
            lines.append(f"{indent}{node['name']} ({child_count} children){desc_preview}")
        else:
            lines.append(f"{indent}{node['name']}{desc_preview}")

        if depth < max_depth:
            children = get_children(node["id"])
            for child in children:
                _walk(child, depth + 1, indent + "  ")

    for root in roots:
        _walk(root, 0, "")

    return "\n".join(lines)


def find_leaf_descendants(node_id: int) -> list[int]:
    """Return all leaf node IDs reachable from node_id."""
    from core.store import get_descendants
    descendants = get_descendants(node_id)
    return [d["id"] for d in descendants if not get_children(d["id"])]


def compute_depth(node_id: int) -> int:
    """Distance from the closest root (0 for root nodes)."""
    from core.store import get_parents

    depth = 0
    current_ids = {node_id}
    visited = set()

    while current_ids:
        # Check if any current node is a root (has no parents)
        next_ids = set()
        for cid in current_ids:
            if cid in visited:
                continue
            visited.add(cid)
            parents = get_parents(cid)
            if not parents:
                return depth
            next_ids.update(p["id"] for p in parents)
        current_ids = next_ids
        depth += 1

    return depth
