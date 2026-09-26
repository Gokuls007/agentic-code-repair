"""Command-line entry point: ``repair-agent``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from repair_agent import __version__
from repair_agent.config import get_settings

app = typer.Typer(
    name="repair-agent",
    help="Autonomous code repair agent with a sandboxed test loop and eval harness.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("config")
def show_config() -> None:
    """Show the effective configuration. Secrets are masked."""
    settings = get_settings()
    dumped = settings.model_dump(mode="json")
    for key in ("anthropic_api_key", "groq_api_key", "github_token"):
        dumped[key] = "set" if getattr(settings, key) else "unset"
    typer.echo(json.dumps(dumped, indent=2))


@app.command()
def ping() -> None:
    """Send one tiny request to the configured LLM to verify credentials (costs a few tokens)."""
    from repair_agent.llm import create_provider
    from repair_agent.llm.base import Message, TextBlock
    from repair_agent.llm.pricing import estimate_cost

    settings = get_settings()
    try:
        provider = create_provider(settings)
    except (RuntimeError, NotImplementedError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    response = provider.complete(
        system="You are a connectivity check. Reply with the single word: pong",
        messages=[Message(role="user", content=[TextBlock(text="ping")])],
        tools=[],
        max_output_tokens=1024,
    )
    cost = estimate_cost(response.usage, settings.llm.model, settings.pricing)
    typer.echo(
        f"model={response.model} stop={response.stop_reason} reply={response.message.text!r}"
    )
    typer.echo(
        f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens} "
        f"latency={response.latency_s:.2f}s cost=" + (f"${cost:.6f}" if cost is not None else "n/a")
    )


@app.command()
def solve(
    task: Annotated[Path, typer.Option(help="Path to a benchmark task YAML file.")],
) -> None:
    """Run the agent on one task. (Implemented in Phase 3.)"""
    typer.echo(f"solve is not implemented yet (Phase 3). Task: {task}", err=True)
    raise typer.Exit(code=2)


@app.command("eval")
def run_eval(
    tasks_dir: Annotated[Path, typer.Option(help="Directory of task YAML files.")] = Path(
        "benchmark/tasks"
    ),
) -> None:
    """Run the benchmark and write a report. (Implemented in Phase 4.)"""
    typer.echo(f"eval is not implemented yet (Phase 4). Tasks: {tasks_dir}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
