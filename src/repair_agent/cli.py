"""Command-line entry point: ``repair-agent``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from repair_agent import __version__
from repair_agent.config import get_settings
from repair_agent.llm.base import LLMError

if TYPE_CHECKING:
    from repair_agent.agent import AgentResult
    from repair_agent.sandbox import DockerSandbox

app = typer.Typer(
    name="repair-agent",
    help="Autonomous code repair agent with a sandboxed test loop and eval harness.",
    no_args_is_help=True,
)
sandbox_app = typer.Typer(help="Manage the Docker sandbox.", no_args_is_help=True)
app.add_typer(sandbox_app, name="sandbox")


def _sandbox() -> DockerSandbox:
    """Sandbox from settings (imported lazily to keep CLI start-up fast)."""
    from repair_agent.sandbox import DockerSandbox

    settings = get_settings()
    return DockerSandbox(settings.sandbox, secret_values=settings.secret_values())


@sandbox_app.command("build")
def sandbox_build() -> None:
    """Build the sandbox image (uses the network once, at build time only)."""
    from repair_agent.sandbox import SandboxError

    sandbox = _sandbox()
    try:
        typer.echo(f"Building {sandbox.settings.image} ...")
        sandbox.build_image()
    except SandboxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Done.")


@sandbox_app.command("cleanup")
def sandbox_cleanup() -> None:
    """Remove sandbox containers left behind by interrupted runs."""
    from repair_agent.sandbox import SandboxError

    try:
        removed = _sandbox().cleanup_orphans()
    except SandboxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Removed {removed} container(s).")


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


def _describe_llm_error(exc: LLMError, model: str, provider: str) -> str:
    """One-line, actionable explanation of a failed LLM call."""
    status = exc.status_code
    if status == 401:
        return f"authentication failed: check {provider.upper()}_API_KEY in .env"
    if status == 403:
        return "permission denied: this API key cannot use the requested resource"
    if status == 404:
        return f"model not found: {model!r} (check REPAIR_LLM__MODEL)"
    if status is None:
        return f"could not reach the API: {exc}"
    return f"API error {status}: {exc}"


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
    try:
        response = provider.complete(
            system="You are a connectivity check. Reply with the single word: pong",
            messages=[Message(role="user", content=[TextBlock(text="ping")])],
            tools=[],
            max_output_tokens=1024,
        )
    except LLMError as exc:
        typer.echo(
            f"error: {_describe_llm_error(exc, settings.llm.model, settings.llm.provider)}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    cost = estimate_cost(response.usage, settings.llm.model, settings.pricing)
    if settings.llm.provider == "groq" and settings.llm.groq_free_tier:
        list_price = f"${cost:.6f}" if cost is not None else "n/a"
        cost_text = f"$0.00 (Groq free tier; list-price equivalent {list_price})"
    else:
        cost_text = f"${cost:.6f}" if cost is not None else "n/a"
    typer.echo(
        f"model={response.model} stop={response.stop_reason} reply={response.message.text!r}"
    )
    typer.echo(
        f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens} "
        f"latency={response.latency_s:.2f}s cost={cost_text}"
    )


@app.command()
def solve(
    task: Annotated[Path, typer.Option(help="Path to a benchmark task YAML file.")],
    keep_workspace: Annotated[
        bool, typer.Option(help="Keep the working copy on disk after the run.")
    ] = False,
) -> None:
    """Run the agent on one task: explore, fix, test in the sandbox, then grade."""
    from repair_agent.agent import load_task, solve_task
    from repair_agent.llm import create_provider
    from repair_agent.sandbox import SandboxError, Workspace
    from repair_agent.tracing import Tracer, new_run_id

    settings = get_settings()
    try:
        task_def = load_task(task)
        provider = create_provider(settings)
        sandbox = _sandbox()
        sandbox.ensure_image()
    except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    run_id = new_run_id()
    workspace = Workspace.from_directory(task_def.repo_dir(), patch=task_def.seed_patch)
    try:
        with Tracer(
            settings.runs_dir, run_id, task_def.id, secret_values=settings.secret_values()
        ) as tracer:
            typer.echo(f"run {run_id}: solving {task_def.id} with {settings.llm.model} ...")
            result = solve_task(
                task_def,
                workspace,
                provider=provider,
                sandbox=sandbox,
                settings=settings,
                tracer=tracer,
                run_id=run_id,
            )
    except SandboxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if keep_workspace:
            typer.echo(f"workspace kept at {workspace.root}")
        else:
            workspace.cleanup()

    _print_result(result, tracer.path)
    raise typer.Exit(code=0 if result.resolved else 1)


def _print_result(r: AgentResult, trace_path: Path) -> None:
    """Human-readable summary of one solve run."""
    cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "n/a"
    temperature = "default" if r.temperature is None else r.temperature
    final = r.final_tests
    tests = (
        f"{final.passed} passed, {final.failed} failed, {final.errors} errors" if final else "n/a"
    )
    rows = [
        ("outcome", f"{r.outcome} (stop: {r.stop_reason})"),
        ("resolved", str(r.resolved)),
        (
            "model",
            f"{r.model_id or r.model} (temperature: {temperature})",
        ),
        ("iterations", str(r.iterations)),
        ("test runs", str(r.test_runs)),
        ("tool calls", ", ".join(f"{k}={v}" for k, v in sorted(r.tool_calls.items()))),
        (
            "tokens",
            f"in {r.tokens_in:,} · out {r.tokens_out:,} · cache read "
            f"{r.cache_read_tokens:,} · cache write {r.cache_write_tokens:,}",
        ),
        ("cost", cost + (f" ({r.cost_note})" if r.cost_note else "")),
        ("time", f"{r.wall_s:.1f}s wall, {r.llm_s:.1f}s in LLM"),
        ("final tests", tests),
        ("tests edited", ", ".join(r.modified_test_files) or "none"),
    ]
    if r.error:
        rows.append(("error", r.error))
    width = max(len(k) for k, _ in rows)
    typer.echo("")
    for key, value in rows:
        typer.echo(f"{key:<{width}}  {value}")
    typer.echo(f"{'trace':<{width}}  {trace_path}")
    if r.finish_summary:
        typer.echo(f"\nagent summary: {r.finish_summary}")
    typer.echo("\n" + (r.diff or "(no changes)"))


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
