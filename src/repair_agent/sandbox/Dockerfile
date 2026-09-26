# Sandbox image for running target-repo tests. Built by `repair-agent sandbox build`.
# Network is only used here, at build time, to install our pinned test tooling;
# containers started from this image run with networking disabled.
FROM python:3.11-slim

RUN pip install --no-cache-dir "pytest==8.3.5" \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin sandbox \
    && mkdir /workspace \
    && chown sandbox:sandbox /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

USER sandbox
WORKDIR /workspace
