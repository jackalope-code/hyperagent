# Copyright (c) Meta Platforms, Inc. and affiliates.
# ============================================================
# task_agent.py — PRIMARY OPTIMIZATION TARGET
#
# The MetaAgent will modify this file to improve task-solving
# performance across generations.  Keep the TaskAgent.forward()
# signature stable; everything else is fair game.
# ============================================================

from agent.base_agent import AgentSystem
from agent.llm import chat_with_agent
from utils.common import extract_json


class TaskAgent(AgentSystem):
    """An agent that solves a given task by querying an LLM.

    This is the agent that the MetaAgent optimises.  The initial
    implementation is deliberately minimal — the MetaAgent should
    improve it over time.
    """

    def __init__(self, model: str = None, log=print):
        # Default to gpt-4o-mini: higher rate limits (150 req/day) than gpt-4o (50/day)
        super().__init__(model=model or "gpt-4o-mini", log=log)

    def forward(self, inputs: dict) -> tuple[str, list]:
        """Solve one task instance.

        Args:
            inputs: Dict with at minimum 'domain' and task-specific fields.

        Returns:
            (prediction, msg_history)
        """
        instruction = f"""You are an agent solving a task.

Task input:
{inputs}

Respond in JSON format:
{{
    "response": <your answer here>
}}"""

        msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="none",
        )

        prediction = "None"
        try:
            last = msg_history[-1]["content"]
            text = _get_text(last)
            data = extract_json(text)
            if data and "response" in data:
                prediction = str(data["response"])
        except Exception:
            pass

        return prediction, msg_history


def _get_text(content) -> str:
    """Extract plain text from an Anthropic content block or string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
            if hasattr(block, "text"):
                return block.text
    return str(content)
