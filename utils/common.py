"""Common utility functions."""

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Extract the last well-formed JSON object from an arbitrary string."""
    if not text:
        return None

    # Try parsing the whole string first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Walk the string and collect all top-level JSON objects
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    candidates.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass

    return candidates[-1] if candidates else None
