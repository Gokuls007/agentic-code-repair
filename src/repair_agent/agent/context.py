"""Keep conversation history bounded by eliding old tool results.

Elision is a history edit, which invalidates the prompt cache from the edit point on,
so it runs in batches (only when the context crosses a threshold), not every turn.
The issue message, assistant turns, and tool-call arguments are never modified.
"""

from __future__ import annotations

import re
from typing import Any

from repair_agent.llm.base import Message, ToolCall, ToolResult

ELIDED_PREFIX = "[elided to save context:"
_ARG_PREVIEW = 60
# First line of edit_file results: "Edited <path> (1 replacement). Lines 3-9 now:" or
# "Created <path> (12 lines)."
_EDITED = re.compile(r"^Edited \S+ \(\d+ replacements?\)\. Lines (\d+)-(\d+) now:")
_CREATED = re.compile(r"^Created \S+ \((\d+) lines\)\.")


def _edit_location(result: ToolResult) -> str:
    """'lines a-b' (or 'created, N lines') taken from an edit_file result, if present."""
    first = result.content.splitlines()[0] if result.content else ""
    if match := _EDITED.match(first):
        return f" lines {match.group(1)}-{match.group(2)}"
    if match := _CREATED.match(first):
        return f" (created, {match.group(1)} lines)"
    return ""


def _describe(call: ToolCall | None, result: ToolResult) -> str:
    if call is None:
        return "tool result"
    args: dict[str, Any] = call.arguments
    if call.name == "read_file":
        lines = f" lines {args.get('start_line', 1)}-{args.get('end_line') or 'end'}"
        return f"read_file {args.get('path', '?')}{lines}"
    if call.name == "edit_file":
        return f"edit_file {args.get('path', '?')}{_edit_location(result)}"
    parts = []
    for key, value in args.items():
        text = str(value)
        if len(text) > _ARG_PREVIEW:
            text = text[:_ARG_PREVIEW] + "..."
        parts.append(f"{key}={text!r}")
    return f"{call.name} {' '.join(parts)}".rstrip()


def stub_for(call: ToolCall | None, result: ToolResult) -> str:
    """Replacement text for an elided tool result."""
    base = f"{ELIDED_PREFIX} {_describe(call, result)}"
    if call is not None and call.name == "run_tests":
        first = result.content.splitlines()[0] if result.content else ""
        return f"{base}. {first}]"
    return f"{base}; call again if needed]"


def elide_old_tool_results(messages: list[Message], keep_recent_turns: int) -> int:
    """Replace tool results older than the last ``keep_recent_turns`` rounds, in place.

    Returns the number of tool results newly elided.
    """
    calls = {c.id: c for m in messages if m.role == "assistant" for c in m.tool_calls}
    result_turns = [
        i
        for i, m in enumerate(messages)
        if m.role == "user" and any(isinstance(b, ToolResult) for b in m.content)
    ]
    old_turns = result_turns[:-keep_recent_turns] if keep_recent_turns else result_turns
    elided = 0
    for index in old_turns:
        message = messages[index]
        new_content = []
        for block in message.content:
            if isinstance(block, ToolResult) and not block.content.startswith(ELIDED_PREFIX):
                block = block.model_copy(
                    update={"content": stub_for(calls.get(block.tool_call_id), block)}
                )
                elided += 1
            new_content.append(block)
        messages[index] = message.model_copy(update={"content": new_content})
    return elided
