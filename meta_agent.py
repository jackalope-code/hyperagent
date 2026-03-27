# Copyright (c) Meta Platforms, Inc. and affiliates.
# ============================================================
# meta_agent.py — SELF-REFERENTIAL META AGENT
#
# This file is itself a target for self-modification.  The MetaAgent
# can edit its own prompting strategy, tool usage patterns, and
# improvement heuristics.  Meta-level improvements accumulate across
# generations and transfer across domains.
# ============================================================

import json
import os
from pathlib import Path
from typing import Optional

from agent.base_agent import AgentSystem
from agent.llm import chat_with_agent


class MetaAgent(AgentSystem):
    """A self-referential agent that improves the codebase, including itself.

    Can modify any file in the repository:
      - task_agent.py     → improve task-solving strategy
      - meta_agent.py     → improve self-improvement strategy (this file!)
      - agent/llm.py      → improve tool infrastructure
      - utils/archive.py  → improve parent selection
      - domains/          → improve evaluation harnesses

    The key property of a Hyperagent: the meta-level modification
    procedure is itself editable, enabling metacognitive self-improvement.
    """

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int = None,
        domain_name: Optional[str] = None,
    ) -> list:
        """Analyse past performance and modify the codebase.

        Args:
            repo_path:       Absolute path to the repository root.
            eval_path:       Path to the previous generation's output directory
                             (contains scores.json and other eval artifacts).
            iterations_left: Number of remaining evolution generations.
            domain_name:     Active evaluation domain (used to pre-load domain file).

        Returns:
            The full conversation message history.
        """
        eval_summary = _load_eval_summary(eval_path)
        codebase_context = _load_codebase_context(repo_path, domain_name)

        instruction = f"""You are a MetaAgent — a self-referential self-improving AI.

## Context
Repository: {repo_path}
Iterations remaining: {iterations_left if iterations_left is not None else "unknown"}

## Previous generation evaluation results
{eval_summary}

## Current codebase (key files pre-loaded — use read_file only for files not shown below)
{codebase_context}

## Your goal
Improve the AI agent system so it scores higher on the evaluation domain.

## What you can do
1. Study the pre-loaded files above to understand the current strategy.
2. Review the evaluation results to identify specific failure patterns.
3. Edit files with write_file or replace_in_file to implement improvements.
4. Use read_file / list_dir only for files not shown above.

## What you should know
- You CAN modify meta_agent.py (this very file) to improve how future
  improvements are generated.  Meta-level improvements accumulate.
- You CAN modify task_agent.py to improve task performance directly.
- You CAN modify agent/llm.py, utils/, or domains/ if beneficial.
- Evaluation runs automatically after you finish — do not try to run it.
- Make targeted, high-impact changes based on evidence from the eval results.
- If eval results show specific failure patterns, address them directly.

Study the pre-loaded files and evaluation results above, then make your improvements."""

        msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="all",
            working_dir=repo_path,
        )

        return msg_history


def _load_codebase_context(repo_path: str, domain_name: Optional[str] = None) -> str:
    """Read key source files and return them formatted for the LLM prompt."""
    root = Path(repo_path)
    # Only pre-load task_agent.py and the domain file — not meta_agent.py itself
    # (it's large and adds too many tokens; the agent can read_file it if needed)
    files_to_load: list[Path] = [root / "task_agent.py"]

    # Add the active domain file if known
    if domain_name:
        domain_file = root / "domains" / domain_name / "domain.py"
        if domain_file.exists():
            files_to_load.append(domain_file)

    _MAX_FILE_CHARS = 2500  # keep prompt compact within 8k-token limit

    sections: list[str] = []
    for fpath in files_to_load:
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if len(content) > _MAX_FILE_CHARS:
                    content = content[:_MAX_FILE_CHARS] + "\n# ...(truncated)"
                rel = fpath.relative_to(root)
                sections.append(f"### {rel}\n```python\n{content}\n```")
            except Exception as exc:
                sections.append(f"### {fpath.name}\n(Could not read: {exc})")

    return "\n\n".join(sections) if sections else "(No files pre-loaded)"


def _load_eval_summary(eval_path: str) -> str:
    """Build a human-readable summary of evaluation results for the MetaAgent."""
    if not eval_path or not os.path.exists(eval_path):
        return "No previous evaluation results available (first generation)."

    lines: list[str] = []
    for fname in ("scores.json", "metadata.json", "results.json"):
        fpath = Path(eval_path) / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                lines.append(f"### {fname}\n```json\n{json.dumps(data, indent=2)}\n```")
            except Exception:
                pass

    if not lines:
        try:
            files = [f.name for f in Path(eval_path).iterdir() if f.is_file()]
            lines.append(f"Eval directory: {eval_path}\nFiles: {files}")
        except Exception:
            lines.append(f"Eval directory exists at {eval_path} but is unreadable.")

    return "\n\n".join(lines)
