"""Archive management and score-proportional parent selection.

The archive is a JSONL file where each line is a JSON object representing one
generation node:

    {
        "id":          int | "initial",
        "parent_id":   int | "initial" | null,
        "score":       float | null,
        "patch_file":  str | null,        # path to incremental diff
        "valid_parent": bool
    }

Patch files record only the *incremental* diff from the parent's state to this
node's state.  To reconstruct generation N, apply the full lineage in order.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional


def load_archive(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_archive(path: str, archive: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for node in archive:
            f.write(json.dumps(node) + "\n")


def add_node(
    archive: list[dict],
    archive_path: str,
    node_id,
    parent_id=None,
    score: Optional[float] = None,
    patch_file: Optional[str] = None,
) -> list[dict]:
    node = {
        "id": node_id,
        "parent_id": parent_id,
        "score": score,
        "patch_file": patch_file,
        "valid_parent": True,
    }
    archive.append(node)
    save_archive(archive_path, archive)
    return archive


def update_node(archive: list[dict], archive_path: str, node_id, **kwargs) -> None:
    for node in archive:
        if node["id"] == node_id:
            node.update(kwargs)
    save_archive(archive_path, archive)


def get_lineage_patches(archive: list[dict], node_id) -> list[str]:
    """Return ordered list of patch file paths from root to node_id."""
    nodes_by_id = {n["id"]: n for n in archive}
    path: list[str] = []
    current = nodes_by_id.get(node_id)
    while current is not None:
        if current.get("patch_file"):
            path.insert(0, current["patch_file"])
        current = nodes_by_id.get(current.get("parent_id"))
    return path


def select_parent(archive: list[dict], method: str = "score_prop"):
    """Choose a parent node id using the specified selection strategy."""
    valid = [n for n in archive if n.get("valid_parent", True)]
    if not valid:
        return None
    scored = [n for n in valid if n.get("score") is not None]
    candidates = scored if scored else valid

    if method == "best":
        return max(candidates, key=lambda n: n.get("score", 0.0))["id"]
    if method == "latest":
        return candidates[-1]["id"]
    if method == "random":
        return random.choice(candidates)["id"]

    # score_prop: sample proportional to score (default)
    scores = [max(0.0, n.get("score", 0.0)) for n in candidates]
    total = sum(scores)
    weights = (
        [s / total for s in scores]
        if total > 0
        else [1.0 / len(candidates)] * len(candidates)
    )
    return random.choices(candidates, weights=weights)[0]["id"]
