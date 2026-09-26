"""Docker sandbox: the only place target-repo code ever executes.

Each :meth:`DockerSandbox.run` call creates a fresh container, copies a snapshot of the
host workspace into it as a tar archive (no bind mounts, so no Windows path translation),
runs one command with networking disabled as a non-root user under CPU/memory/PID limits,
collects output and artifacts, and always removes the container.
"""

from __future__ import annotations

import contextlib
import io
import os
import tarfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import docker
import docker.errors
import requests
from pydantic import BaseModel, Field

from repair_agent.config import SandboxSettings
from repair_agent.sandbox.junit import TestReport, parse_junit
from repair_agent.tracing import Redactor

LABEL = "repair-agent.sandbox"
SANDBOX_UID = 1000
CONTAINER_WORKDIR = "/workspace"
JUNIT_PATH = "/tmp/junit.xml"
DOCKERFILE_DIR = Path(__file__).parent

# Never shipped into the container: VCS data, caches, local virtualenvs.
EXCLUDED_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")

PYTEST_ARGS = [
    "python",
    "-m",
    "pytest",
    "-q",
    "-rfE",
    "--tb=short",
    "-p",
    "no:cacheprovider",
    f"--junitxml={JUNIT_PATH}",
    "-o",
    "junit_family=xunit1",  # xunit1 records each test's file path
]


class SandboxError(RuntimeError):
    """The sandbox itself failed (Docker unreachable, image missing, API error)."""


class ExecResult(BaseModel):
    """Outcome of one command run in a fresh container."""

    exit_code: int | None
    output: str
    timed_out: bool = False
    oom: bool = False
    duration_s: float
    artifacts: dict[str, bytes] = Field(default_factory=dict)


def build_workspace_tar(root: Path) -> bytes:
    """Pack ``root`` as ``workspace/...`` with POSIX names, uid 1000, and explicit modes.

    Symlinks are skipped: they may point outside the workspace, and the target repos
    do not need them. File bytes are copied unchanged, so line endings are preserved.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.addfile(_tar_entry("workspace", tarfile.DIRTYPE, 0o755))
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in EXCLUDED_DIRS and not os.path.islink(os.path.join(dirpath, d))
            )
            rel_dir = PurePosixPath(Path(dirpath).relative_to(root).as_posix())
            for d in dirnames:
                tar.addfile(_tar_entry(_arcname(rel_dir / d), tarfile.DIRTYPE, 0o755))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if name.endswith(EXCLUDED_SUFFIXES) or path.is_symlink():
                    continue
                data = path.read_bytes()
                info = _tar_entry(_arcname(rel_dir / name), tarfile.REGTYPE, 0o644)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _arcname(rel: PurePosixPath) -> str:
    return str(PurePosixPath("workspace") / rel)


def _tar_entry(name: str, kind: bytes, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.uid = info.gid = SANDBOX_UID
    info.uname = info.gname = "sandbox"
    info.mtime = int(time.time())
    return info


def _extract_single_file(chunks: Iterable[bytes]) -> bytes | None:
    """Return the first regular file's bytes from a tar stream (``get_archive`` output)."""
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as tar:
        for member in tar.getmembers():
            if member.isfile():
                fh = tar.extractfile(member)
                return fh.read() if fh else None
    return None


class DockerSandbox:
    """Runs commands against a workspace snapshot in locked-down, throwaway containers."""

    def __init__(
        self,
        settings: SandboxSettings,
        *,
        client: Any | None = None,
        secret_values: list[str] | None = None,
    ):
        """``client`` may be injected (tests); otherwise one is created lazily from env."""
        self.settings = settings
        self._client = client
        self._redactor = Redactor(secret_values)

    @property
    def client(self) -> Any:
        """The Docker client, created on first use."""
        if self._client is None:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except (docker.errors.DockerException, requests.RequestException) as exc:
                raise SandboxError(
                    "Docker is not reachable. Start Docker Desktop and try again."
                ) from exc
        return self._client

    def container_kwargs(self, argv: Sequence[str]) -> dict[str, Any]:
        """The exact ``containers.create`` arguments. Security settings live here."""
        s = self.settings
        memory = f"{s.memory_mb}m"
        return {
            "image": s.image,
            "command": list(argv),
            "user": f"{SANDBOX_UID}:{SANDBOX_UID}",
            "working_dir": CONTAINER_WORKDIR,
            "network_mode": "none",
            "mem_limit": memory,
            "memswap_limit": memory,  # equal to mem_limit: no swap
            "nano_cpus": int(s.cpus * 1_000_000_000),
            "pids_limit": s.pids_limit,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": f"{CONTAINER_WORKDIR}/src:{CONTAINER_WORKDIR}",
                "HOME": "/tmp",
            },
            "labels": {LABEL: "1"},
        }

    def run(
        self,
        workspace_root: Path,
        argv: Sequence[str],
        *,
        timeout_s: float | None = None,
        collect: Sequence[str] = (),
    ) -> ExecResult:
        """Run ``argv`` in a fresh container holding a copy of ``workspace_root``.

        ``collect`` lists container file paths to return as artifacts (missing ones are
        skipped). Raises SandboxError only for sandbox failures, never for a non-zero
        exit code of the command itself.
        """
        timeout = timeout_s or self.settings.test_timeout_s
        archive = build_workspace_tar(Path(workspace_root))
        try:
            container = self.client.containers.create(**self.container_kwargs(argv))
        except docker.errors.ImageNotFound as exc:
            raise SandboxError(
                f"Sandbox image {self.settings.image!r} not found. "
                "Run `repair-agent sandbox build`."
            ) from exc
        except docker.errors.APIError as exc:
            raise SandboxError(f"Could not create sandbox container: {exc}") from exc

        started = time.perf_counter()
        try:
            container.put_archive("/", archive)
            container.start()
            timed_out, exit_code = self._wait(container, timeout)
            duration = time.perf_counter() - started
            oom = False
            if not timed_out:
                container.reload()
                oom = bool(container.attrs.get("State", {}).get("OOMKilled"))
            output = self._logs(container)
            artifacts = {} if timed_out else self._collect(container, collect)
        except docker.errors.APIError as exc:
            raise SandboxError(f"Sandbox command failed to run: {exc}") from exc
        finally:
            self._remove(container)

        return ExecResult(
            exit_code=None if timed_out else exit_code,
            output=output,
            timed_out=timed_out,
            oom=oom,
            duration_s=round(duration, 3),
            artifacts=artifacts,
        )

    def run_pytest(
        self,
        workspace_root: Path,
        selectors: Sequence[str] = (),
        *,
        timeout_s: float | None = None,
    ) -> TestReport:
        """Run pytest (optionally on specific node ids/paths) and parse per-test outcomes."""
        result = self.run(
            workspace_root,
            [*PYTEST_ARGS, *selectors],
            timeout_s=timeout_s,
            collect=[JUNIT_PATH],
        )
        junit = result.artifacts.get(JUNIT_PATH, b"")
        return TestReport(
            outcomes=parse_junit(junit),
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            timed_out=result.timed_out,
            oom=result.oom,
            output=result.output,
        )

    def image_exists(self) -> bool:
        """True if the configured sandbox image is present locally."""
        try:
            self.client.images.get(self.settings.image)
        except docker.errors.ImageNotFound:
            return False
        return True

    def build_image(self) -> None:
        """Build the sandbox image from the bundled Dockerfile."""
        try:
            self.client.images.build(
                path=str(DOCKERFILE_DIR), dockerfile="Dockerfile", tag=self.settings.image, rm=True
            )
        except (docker.errors.BuildError, docker.errors.APIError) as exc:
            raise SandboxError(f"Building {self.settings.image!r} failed: {exc}") from exc

    def ensure_image(self) -> None:
        """Build the sandbox image if it is missing."""
        if not self.image_exists():
            self.build_image()

    def cleanup_orphans(self) -> int:
        """Remove any sandbox containers left behind (e.g. by a killed process)."""
        containers = self.client.containers.list(all=True, filters={"label": LABEL})
        for container in containers:
            self._remove(container)
        return len(containers)

    def _wait(self, container: Any, timeout: float) -> tuple[bool, int | None]:
        try:
            status = container.wait(timeout=timeout)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            with contextlib.suppress(docker.errors.APIError):  # exited before the kill
                container.kill()
            return True, None
        return False, int(status.get("StatusCode", -1))

    def _logs(self, container: Any) -> str:
        raw: bytes = container.logs(stdout=True, stderr=True)
        limit = self.settings.max_log_bytes
        text = raw[-limit:].decode("utf-8", errors="replace")
        if len(raw) > limit:
            text = f"[... first {len(raw) - limit} bytes of output dropped ...]\n" + text
        return self._redactor.scrub(text.replace("\r\n", "\n"))

    def _collect(self, container: Any, paths: Sequence[str]) -> dict[str, bytes]:
        artifacts: dict[str, bytes] = {}
        for path in paths:
            try:
                chunks, _stat = container.get_archive(path)
            except docker.errors.NotFound:
                continue
            data = _extract_single_file(chunks)
            if data is not None:
                artifacts[path] = data
        return artifacts

    @staticmethod
    def _remove(container: Any) -> None:
        with contextlib.suppress(docker.errors.NotFound):
            container.remove(force=True)
