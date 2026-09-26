"""Per-repo sandbox images with third-party dependencies baked in at build time.

Test containers never have network access, so a repo that needs packages gets its own
image: ``FROM`` the base sandbox image, with ``requirements.txt`` installed as root using
``--require-hashes --only-binary=:all:`` (pinned, hash-checked wheels only, so no package
build code runs). The tag is a content hash, so images are rebuilt only when the
requirements, the recipe, or the base image change.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
from pathlib import Path

import docker.errors

from repair_agent.sandbox.docker import DockerSandbox, SandboxError

REQUIREMENTS = "requirements.txt"

_RECIPE = """\
FROM {base}
USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.txt \\
    && rm /tmp/requirements.txt
USER sandbox
WORKDIR /workspace
"""


def dockerfile_for(base_image: str) -> str:
    """The Dockerfile used for dependency images built on ``base_image``."""
    return _RECIPE.format(base=base_image)


def repo_image_tag(repo_dir: Path, base_image: str) -> str | None:
    """Content-addressed tag for ``repo_dir``'s dependency image, or None if it has none."""
    requirements = Path(repo_dir) / REQUIREMENTS
    if not requirements.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(dockerfile_for(base_image).encode("utf-8"))
    digest.update(requirements.read_bytes())
    name = re.sub(r"[^a-z0-9]+", "-", Path(repo_dir).name.lower()).strip("-")
    return f"repair-agent-sandbox-{name}:{digest.hexdigest()[:12]}"


def _build_context(base_image: str, requirements: bytes) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in (
            ("Dockerfile", dockerfile_for(base_image).encode("utf-8")),
            (REQUIREMENTS, requirements),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def sandbox_for_repo(sandbox: DockerSandbox, repo_dir: Path) -> DockerSandbox:
    """A sandbox using ``repo_dir``'s dependency image (built if missing), or ``sandbox``."""
    sandbox.ensure_image()
    tag = repo_image_tag(repo_dir, sandbox.settings.image)
    if tag is None:
        return sandbox
    derived = sandbox.with_image(tag)
    if not derived.image_exists():
        requirements = (Path(repo_dir) / REQUIREMENTS).read_bytes()
        try:
            derived.client.images.build(
                fileobj=_build_context(sandbox.settings.image, requirements),
                custom_context=True,
                tag=tag,
                rm=True,
            )
        except (docker.errors.BuildError, docker.errors.APIError) as exc:
            raise SandboxError(f"Building dependency image {tag!r} failed: {exc}") from exc
    return derived


def task_sandbox(
    sandbox: DockerSandbox, image_override: str | None, repo_dir: Path
) -> DockerSandbox:
    """The sandbox a task runs in: explicit image override, else the repo's dependency image."""
    if image_override:
        return sandbox.with_image(image_override)
    return sandbox_for_repo(sandbox, repo_dir)
