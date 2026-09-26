from __future__ import annotations

from pathlib import Path

import pytest

from repair_agent.sandbox.workspace import Workspace, WorkspaceError, git

SEED_PATCH = """\
--- a/src/calc/ops.py
+++ b/src/calc/ops.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a + b
+    return a - b
 \n"""


def test_clone_is_independent_of_source(calc_source: Path, tmp_path: Path) -> None:
    with Workspace.create(calc_source, parent_dir=tmp_path) as ws:
        assert (ws.root / "src/calc/stats.py").is_file()
        assert ws.root != calc_source.resolve()
        assert ws.diff() == ""
    assert not ws.root.exists()


def test_checkout_base_commit(calc_source: Path, tmp_path: Path) -> None:
    first = git(calc_source, "rev-parse", "HEAD").strip()
    (calc_source / "later.txt").write_text("later\n", encoding="utf-8")
    git(calc_source, "add", "-A")
    git(calc_source, "-c", "user.name=t", "-c", "user.email=t@l", "commit", "-qm", "later")
    with Workspace.create(calc_source, base_commit=first, parent_dir=tmp_path) as ws:
        assert not (ws.root / "later.txt").exists()


def test_seed_patch_is_applied_and_committed(calc_source: Path, tmp_path: Path) -> None:
    with Workspace.create(calc_source, patch=SEED_PATCH, parent_dir=tmp_path) as ws:
        assert b"return a - b" in (ws.root / "src/calc/ops.py").read_bytes()
        assert ws.diff() == ""  # the seeded bug is part of the baseline, not the agent's diff


def test_diff_shows_edits_and_new_files(workspace: Workspace) -> None:
    ops = workspace.root / "src/calc/ops.py"
    ops.write_bytes(ops.read_bytes().replace(b"a + b", b"b + a"))
    (workspace.root / "src/calc/extra.py").write_bytes(b"Z = 1\n")
    diff = workspace.diff()
    assert "-    return a + b" in diff
    assert "+    return b + a" in diff
    assert "+++ b/src/calc/extra.py" in diff


def test_checkout_is_byte_exact_despite_global_autocrlf(calc_source: Path, tmp_path: Path) -> None:
    with Workspace.create(calc_source, parent_dir=tmp_path) as ws:
        assert b"\r\n" not in (ws.root / "src/calc/ops.py").read_bytes()


def test_bad_patch_raises_and_cleans_up(calc_source: Path, tmp_path: Path) -> None:
    parent = tmp_path / "ws"
    parent.mkdir()
    with pytest.raises(WorkspaceError, match="git apply"):
        Workspace.create(calc_source, patch="not a patch\n", parent_dir=parent)
    assert list(parent.iterdir()) == []
