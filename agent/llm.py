"""LLM interaction module with tool support for the HyperAgents system.

Provides chat_with_agent(), which runs an agentic loop against the GitHub Models
API (OpenAI-compatible). When tools_available='all', the agent has access to file
I/O and Python execution tools, enabling the MetaAgent to explore and modify the
codebase.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM  (OpenAI function-calling format)
# ---------------------------------------------------------------------------
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Optionally limit to a line range "
                "(start_line / end_line are 1-indexed, inclusive)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "start_line": {"type": "integer", "description": "First line (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Last line (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating it (and any parent directories) "
                "if it does not exist, or fully overwriting it if it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Replace the first occurrence of old_str with new_str inside a file. "
                "old_str must appear exactly once; be more specific if it matches multiple times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "Exact string to find."},
                    "new_str": {"type": "string", "description": "Replacement string."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the files and subdirectories inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Defaults to working directory.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python code snippet in a subprocess (in the working directory) "
                "and return its stdout + stderr. Use for quick calculations, file inspection, "
                "or verifying edits — not for long-running jobs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to run."},
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to wait (default 30).",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _resolve(path: str, working_dir: str) -> Path:
    p = Path(path)
    return (Path(working_dir) / p).resolve() if not p.is_absolute() else p.resolve()


def _tool_read_file(inputs: dict, working_dir: str) -> str:
    path = _resolve(inputs["path"], working_dir)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    start = max(0, inputs.get("start_line", 1) - 1)
    end = inputs.get("end_line", len(lines))
    selected = lines[start:end]
    return f"# {path} (lines {start + 1}-{start + len(selected)})\n" + "".join(selected)


def _tool_write_file(inputs: dict, working_dir: str) -> str:
    path = _resolve(inputs["path"], working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inputs["content"], encoding="utf-8")
    return f"Wrote {len(inputs['content'])} chars to {path}"


def _tool_replace_in_file(inputs: dict, working_dir: str) -> str:
    path = _resolve(inputs["path"], working_dir)
    text = path.read_text(encoding="utf-8")
    old_str = inputs["old_str"]
    count = text.count(old_str)
    if count == 0:
        return f"Error: old_str not found in {path}"
    if count > 1:
        return f"Error: old_str appears {count} times in {path}; be more specific."
    path.write_text(text.replace(old_str, inputs["new_str"], 1), encoding="utf-8")
    return f"Replaced in {path}"


def _tool_list_dir(inputs: dict, working_dir: str) -> str:
    path = _resolve(inputs.get("path", "."), working_dir)
    if not path.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"


def _tool_run_python(inputs: dict, working_dir: str) -> str:
    timeout = inputs.get("timeout", 30)
    result = subprocess.run(
        [sys.executable, "-c", inputs["code"]],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = result.stdout
    if result.returncode != 0:
        out += f"\n--- stderr ---\n{result.stderr}"
    return out or "(no output)"


def _execute_tool(name: str, inputs: dict, working_dir: str, log: Callable) -> str:
    try:
        dispatch = {
            "read_file": _tool_read_file,
            "write_file": _tool_write_file,
            "replace_in_file": _tool_replace_in_file,
            "list_dir": _tool_list_dir,
            "run_python": _tool_run_python,
        }
        if name not in dispatch:
            return f"Unknown tool: {name}"
        return dispatch[name](inputs, working_dir)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


# ---------------------------------------------------------------------------
# GitHub Models / OpenAI client factory
# ---------------------------------------------------------------------------

def _make_client() -> OpenAI:
    token = os.environ.get("GITHUB_MODELS_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_MODELS_TOKEN is not set. "
            "Create a GitHub Personal Access Token (fine-grained: Account permissions -> "
            "Models -> Read-only; classic: no special scope needed) and add it to your .env file."
        )
    endpoint = os.environ.get(
        "GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com"
    )
    return OpenAI(api_key=token, base_url=endpoint)


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def chat_with_agent(
    instruction: str,
    model: str,
    msg_history: list,
    logging: Callable = print,
    tools_available: str = "none",
    working_dir: str = ".",
    max_turns: int = 50,
) -> list:
    """Run an agentic conversation loop against the GitHub Models API.

    Args:
        instruction:     Initial user message.
        model:           Model identifier (e.g. 'gpt-4o').
        msg_history:     Prior conversation (list of role/content dicts).
        logging:         Callable used to emit progress logs.
        tools_available: 'all' enables file/code tools; 'none' for plain chat.
        working_dir:     Working directory for tool execution.
        max_turns:       Maximum tool-use rounds before stopping.

    Returns:
        Updated full message history.
    """
    client = _make_client()
    tools = _TOOLS if tools_available == "all" else []
    messages = list(msg_history) + [{"role": "user", "content": instruction}]
    _large = os.environ.get("USE_LARGER_CONTEXT", "false").strip().lower() == "true"
    _max_tokens = 8096 if _large else 2048
    _tool_result_limit = 8000 if _large else 2000

    for turn in range(max_turns):
        kwargs: dict = dict(model=model, max_tokens=_max_tokens, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required" if turn == 0 else "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        logging(f"[llm turn {turn}] finish_reason={finish_reason}")

        # Persist the assistant message (keep tool_calls if present)
        assistant_msg: dict = {"role": "assistant", "content": choice.message.content or ""}
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]
        messages.append(assistant_msg)

        if finish_reason == "stop":
            break

        if finish_reason == "tool_calls":
            for tc in choice.message.tool_calls:
                name = tc.function.name
                try:
                    inputs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}
                logging(f"  tool={name} inputs={inputs}")
                result = _execute_tool(name, inputs, working_dir, logging)
                logging(f"  -> {result[:300]!r}")
                # Truncate long results to keep message history within token budget
                trimmed = result if len(result) <= _tool_result_limit else result[:_tool_result_limit] + "\n...(truncated)"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": trimmed}
                )

    return messages

