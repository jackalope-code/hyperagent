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
    ) -> list:
        """Analyse past performance and modify the codebase.

        Args:
            repo_path:       Absolute path to the repository root.
            eval_path:       Path to the previous generation's output directory
                             (contains scores.json and other eval artifacts).
            iterations_left: Number of remaining evolution generations.

        Returns:
            The full conversation message history.
        """
        eval_summary = _load_eval_summary(eval_path)

        instruction = f"""You are a MetaAgent — a self-referential self-improving AI.

## Context
Repository: {repo_path}
Iterations remaining: {iterations_left if iterations_left is not None else "unknown"}

## Previous generation evaluation results
{eval_summary}

## Your goal
Improve the AI agent system so it scores higher on the evaluation domain.

## What you can do
1. Explore the repository with list_dir and read_file.
2. Study task_agent.py to understand the current task-solving strategy.
3. Study meta_agent.py (this file) to understand the self-improvement strategy.
4. Review the evaluation results to identify weaknesses.
5. Edit files with write_file or replace_in_file to implement improvements.

## What you should know
- You CAN modify meta_agent.py (this very file) to improve how future
  improvements are generated.  Meta-level improvements accumulate.
- You CAN modify task_agent.py to improve task performance directly.
- You CAN modify agent/llm.py, utils/, or domains/ if beneficial.
- Evaluation runs automatically after you finish — do not try to run it.
- Make targeted, high-impact changes based on evidence from the eval results.
- If eval results show specific failure patterns, address them directly.

Start by listing the repository structure, then read the relevant files."""

        msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="all",
            working_dir=repo_path,
        )

        return msg_history


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
