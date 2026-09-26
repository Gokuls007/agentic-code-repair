from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repair_agent.tracing import REDACTED, EventKind, Redactor, Tracer, new_run_id, read_trace


def test_writes_one_json_line_per_event(tmp_path: Path) -> None:
    with Tracer(tmp_path, "run1", "task-a") as tracer:
        tracer.log(EventKind.TASK_START, {"issue": "fix it"})
        tracer.next_step()
        tracer.log(EventKind.LLM_CALL, {"model": "m"}, tokens_in=100, tokens_out=20)
        tracer.log(EventKind.TOOL_CALL, {"name": "read_file", "args": {"path": "a.py"}})

    path = tmp_path / "run1" / "task-a.jsonl"
    assert tracer.path == path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line) for line in lines)


def test_round_trip_preserves_fields_and_steps(tmp_path: Path) -> None:
    with Tracer(tmp_path, "run1", "t") as tracer:
        tracer.log(EventKind.TASK_START)
        assert tracer.next_step() == 1
        tracer.log(EventKind.THOUGHT, {"text": "look at utils"}, tokens_in=5, tokens_out=6)
        tracer.next_step()
        tracer.log(EventKind.TASK_END, {"success": True})

    events = read_trace(tracer.path)
    assert [e.kind for e in events] == [EventKind.TASK_START, EventKind.THOUGHT, EventKind.TASK_END]
    assert [e.step for e in events] == [0, 1, 2]
    assert events[1].data == {"text": "look at utils"}
    assert (events[1].tokens_in, events[1].tokens_out) == (5, 6)
    assert all(e.timestamp.tzinfo is not None for e in events)


def test_events_are_flushed_before_close(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path, "run1", "t").open()
    try:
        tracer.log(EventKind.TASK_START, {"x": 1})
        assert len(read_trace(tracer.path)) == 1  # readable while still open
    finally:
        tracer.close()


def test_configured_secrets_are_redacted_everywhere(tmp_path: Path) -> None:
    secret = "my-custom-secret-value"
    with Tracer(tmp_path, "run1", "t", secret_values=[secret]) as tracer:
        tracer.log(
            EventKind.TOOL_RESULT,
            {"output": f"token={secret}", "nested": {"list": [secret, "ok"]}},
        )
    raw = tracer.path.read_text(encoding="utf-8")
    assert secret not in raw
    event = read_trace(tracer.path)[0]
    assert event.data["output"] == f"token={REDACTED}"
    assert event.data["nested"]["list"] == [REDACTED, "ok"]


@pytest.mark.parametrize(
    "leak",
    [
        "sk-ant-api03-AbCdEfGhIjKlMnOp",
        "gsk_ABCDEFGHIJKLMNOPQRSTUV123",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "github_pat_11ABCDEFG0123456789_abcdefghij",
    ],
)
def test_credential_shaped_strings_are_redacted_without_config(leak: str) -> None:
    assert Redactor().scrub(f"env dump: KEY={leak} done") == f"env dump: KEY={REDACTED} done"


def test_redactor_leaves_non_strings_alone() -> None:
    assert Redactor(["s3cret"]).scrub({"n": 3, "b": True, "none": None}) == {
        "n": 3,
        "b": True,
        "none": None,
    }


def test_log_requires_open_tracer(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not open"):
        Tracer(tmp_path, "run1", "t").log(EventKind.TASK_START)


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".hidden"])
def test_rejects_unsafe_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        Tracer(tmp_path, "run1", bad)


def test_new_run_id_is_sortable_and_unique() -> None:
    fixed = datetime(2026, 9, 25, 14, 15, 3, tzinfo=UTC)
    rid = new_run_id(fixed)
    assert rid.startswith("20260925-141503-")
    assert new_run_id(fixed) != rid
