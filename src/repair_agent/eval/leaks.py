"""Check that a task's issue describes symptoms only.

An issue must not reveal where the bug is or what the fix is. We flag:

- the path (or file name) of any file the seed patch touches;
- any changed line of code from the seed patch (stripped, 8+ characters);
- phrases that point at the cause rather than the symptom.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from repair_agent.agent.task import Task

HINT_PHRASES = (
    "off-by-one",
    "off by one",
    "the bug is",
    "the problem is",
    "root cause",
    "should use",
    "should call",
    "instead of calling",
)
MIN_LINE_CHARS = 8

_PLUS_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def patched_files(patch: str) -> list[str]:
    """Paths the unified diff modifies (from its ``+++ b/...`` headers)."""
    return sorted(set(_PLUS_FILE.findall(patch)))


def changed_lines(patch: str) -> list[str]:
    """Added/removed code lines (stripped), excluding diff headers."""
    lines = []
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line[:1] in "+-":
            stripped = line[1:].strip()
            # Import lines say nothing about the bug and naturally appear in examples.
            if len(stripped) >= MIN_LINE_CHARS and not stripped.startswith(("import ", "from ")):
                lines.append(stripped)
    return lines


def find_leaks(task: Task) -> list[str]:
    """Human-readable reasons the issue text gives the bug away (empty if clean)."""
    text = f"{task.issue.title}\n{task.issue.body}"
    lowered = text.lower()
    problems = []
    patch = task.seed_patch or ""
    for path in patched_files(patch):
        for needle in (path, PurePosixPath(path).name):
            if needle in text:
                problems.append(f"issue mentions patched file {needle!r}")
                break
    for line in changed_lines(patch):
        if line in text:
            problems.append(f"issue contains a changed line of code: {line!r}")
    for phrase in HINT_PHRASES:
        if phrase in lowered:
            problems.append(f"issue uses a cause/fix hint: {phrase!r}")
    return problems
