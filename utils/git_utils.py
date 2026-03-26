"""Git utilities for version-tracking agent modifications.

Each generation captures the MetaAgent's changes as a unified diff (patch file).
The generate loop applies lineage patches to reconstruct any ancestor state.
"""

import os
import subprocess
from pathlib import Path


def _run(args: list[str], cwd: str, input_text: str = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        input=input_text,
    )


def init_repo(repo_path: str) -> None:
    """Ensure repo_path is a git repo with at least one commit."""
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        _run(["git", "init"], repo_path)
        _run(["git", "config", "user.email", "hyperagent@localhost"], repo_path)
        _run(["git", "config", "user.name", "HyperAgent"], repo_path)

    result = _run(["git", "log", "--oneline", "-1"], repo_path)
    if not result.stdout.strip():
        _run(["git", "add", "-A"], repo_path)
        _run(["git", "commit", "-m", "Initial HyperAgents commit"], repo_path)


def get_head_commit(repo_path: str) -> str:
    return _run(["git", "rev-parse", "HEAD"], repo_path).stdout.strip()


def get_current_diff(repo_path: str, base_commit: str) -> str:
    """Return a unified diff of all changes since base_commit."""
    return _run(["git", "diff", base_commit, "--", "."], repo_path).stdout


def apply_patch(repo_path: str, patch_path: str) -> bool:
    """Apply a patch file. Returns True on success."""
    patch = Path(patch_path).read_text(encoding="utf-8")
    if not patch.strip():
        return True
    result = _run(
        ["git", "apply", "--whitespace=fix", "--allow-empty"],
        repo_path,
        input_text=patch,
    )
    return result.returncode == 0


def reset_to_commit(repo_path: str, commit: str) -> None:
    """Hard-reset the working tree to commit, discarding all changes."""
    _run(["git", "reset", "--hard", commit], repo_path)
    _run(["git", "clean", "-fd"], repo_path)


def stage_and_commit(repo_path: str, message: str) -> str:
    """Stage everything and create a commit. Returns new HEAD hash."""
    _run(["git", "add", "-A"], repo_path)
    _run(["git", "commit", "-m", message, "--allow-empty"], repo_path)
    return get_head_commit(repo_path)
