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
    from repair_agent.agent.task import Task
    from repair_agent.sandbox import DockerSandbox

app = typer.Typer(
    name="repair-agent",
    help="Autonomous code repair agent with a sandboxed test loop and eval harness.",
    no_args_is_help=True,
)
sandbox_app = typer.Typer(help="Manage the Docker sandbox.", no_args_is_help=True)
app.add_typer(sandbox_app, name="sandbox")
benchmark_app = typer.Typer(help="Benchmark tasks.", no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")

DEFAULT_TASKS_DIR = Path("benchmark/tasks")


@benchmark_app.command("validate")
def benchmark_validate(
    tasks_dir: Annotated[Path, typer.Option(help="Directory of task YAML files.")] = (
        DEFAULT_TASKS_DIR
    ),
    filter_: Annotated[str, typer.Option("--filter", help="Glob on task ids.")] = "*",
    repeat: Annotated[int, typer.Option(help="Clean-suite runs for the flakiness check.")] = 2,
    fill_tests: Annotated[
        bool, typer.Option(help="Write measured FAIL_TO_PASS/PASS_TO_PASS into the YAML.")
    ] = False,
) -> None:
    """Prove each task is well-formed, with no LLM: bug fails the right tests, fix passes."""
    from repair_agent.agent.task import load_task, load_tasks
    from repair_agent.eval.validate import ValidationCache, validate_task, write_test_lists
    from repair_agent.sandbox import SandboxError

    settings = get_settings()
    tasks = load_tasks(tasks_dir, filter_)
    if not tasks:
        typer.echo(f"error: no tasks match {filter_!r} in {tasks_dir}", err=True)
        raise typer.Exit(code=1)
    sandbox = _sandbox()
    try:
        sandbox.ensure_image()
    except SandboxError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    cache = ValidationCache(settings.runs_dir / "validation.json")

    failures = 0
    for task in tasks:
        result = validate_task(task, sandbox, repeat=repeat)
        if fill_tests and result.measured_fail_to_pass:
            write_test_lists(task, result.measured_fail_to_pass, result.measured_pass_to_pass)
            result = validate_task(load_task(task.source_path), sandbox, repeat=repeat)
        cache.put(result)
        status = "ok " if result.ok else "BAD"
        typer.echo(
            f"{status} {task.id:<22} {task.bug_type:<18} {task.difficulty:<6} "
            f"f2p={len(result.measured_fail_to_pass):<2} p2p={len(result.measured_pass_to_pass):<3}"
            f" tests={result.clean_tests:<3} {result.duration_s:5.1f}s"
        )
        for problem in result.problems:
            typer.echo(f"      - {problem}")
        failures += not result.ok
    typer.echo(f"\n{len(tasks) - failures}/{len(tasks)} tasks valid")
    raise typer.Exit(code=1 if failures else 0)


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
    from repair_agent.eval.power import keep_awake
    from repair_agent.llm import create_provider
    from repair_agent.sandbox import SandboxError, Workspace
    from repair_agent.sandbox.images import task_sandbox
    from repair_agent.tracing import Tracer, new_run_id

    settings = get_settings()
    try:
        task_def = load_task(task)
        provider = create_provider(settings)
        sandbox = task_sandbox(_sandbox(), task_def.image, task_def.repo_dir())
    except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    run_id = new_run_id()
    workspace = Workspace.from_directory(task_def.repo_dir(), patch=task_def.seed_patch)
    try:
        with (
            keep_awake(),
            Tracer(
                settings.runs_dir, run_id, task_def.id, secret_values=settings.secret_values()
            ) as tracer,
        ):
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
    tasks_dir: Annotated[Path, typer.Option(help="Directory of task YAML files.")] = (
        DEFAULT_TASKS_DIR
    ),
    runs: Annotated[int, typer.Option(min=1, help="Attempts per task.")] = 3,
    filter_: Annotated[str, typer.Option("--filter", help="Glob on task ids.")] = "*",
    resume: Annotated[str | None, typer.Option(help="Eval id to resume.")] = None,
    force: Annotated[bool, typer.Option(help="Resume even if config/tasks changed.")] = False,
    parallel: Annotated[int, typer.Option(min=1, help="Concurrent attempts.")] = 1,
    skip_validation: Annotated[
        bool, typer.Option(help="Skip the benchmark validation gate (debugging only).")
    ] = False,
) -> None:
    """Run every task N times (resumable) and write report.md / report.json."""
    from repair_agent.agent.task import load_tasks
    from repair_agent.eval.metrics import compute_metrics
    from repair_agent.eval.power import keep_awake
    from repair_agent.eval.report import write_report
    from repair_agent.eval.runner import (
        EvalRunner,
        build_manifest,
        load_attempts,
        load_manifest,
        manifest_wall_limit,
        new_eval_id,
        write_manifest,
    )
    from repair_agent.eval.validate import ValidationCache, ensure_validated
    from repair_agent.llm import create_provider
    from repair_agent.sandbox import SandboxError
    from repair_agent.sandbox.images import task_sandbox

    settings = get_settings()
    tasks = load_tasks(tasks_dir, filter_)
    if not tasks:
        typer.echo(f"error: no tasks match {filter_!r} in {tasks_dir}", err=True)
        raise typer.Exit(code=1)
    try:
        provider = create_provider(settings)
        base = _sandbox()
        base.ensure_image()
    except (RuntimeError, NotImplementedError, SandboxError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not skip_validation:
        typer.echo(f"validating {len(tasks)} task(s) (cached results are reused) ...")
        cache = ValidationCache(settings.runs_dir / "validation.json")
        bad = [v for v in ensure_validated(tasks, base, cache) if not v.ok]
        if bad:
            for v in bad:
                typer.echo(f"invalid task {v.task_id}: {'; '.join(v.problems)}", err=True)
            typer.echo(
                "fix these (see `repair-agent benchmark validate`) before evaluating.", err=True
            )
            raise typer.Exit(code=1)

    eval_id = resume or new_eval_id()
    eval_dir = settings.runs_dir / eval_id
    manifest = build_manifest(eval_id, settings, provider.name, tasks, runs)
    if resume:
        if not (eval_dir / "manifest.json").is_file():
            typer.echo(f"error: no eval {resume!r} under {settings.runs_dir}", err=True)
            raise typer.Exit(code=1)
        previous = load_manifest(eval_dir)
        problems = previous.compatible_with(manifest)
        if problems and not force:
            typer.echo(
                f"error: cannot resume {resume}: {'; '.join(problems)} (use --force)", err=True
            )
            raise typer.Exit(code=1)
        manifest = previous
    else:
        write_manifest(eval_dir, manifest)

    sandboxes: dict[str, DockerSandbox] = {}

    def sandbox_for(task: Task) -> DockerSandbox:
        """One sandbox (and dependency image) per repo, built on first use."""
        key = task.image or task.repo
        if key not in sandboxes:
            sandboxes[key] = task_sandbox(base, task.image, task.repo_dir())
        return sandboxes[key]

    runner = EvalRunner(
        settings=settings,
        tasks=tasks,
        eval_dir=eval_dir,
        runs=runs,
        provider=provider,
        sandbox_for=sandbox_for,
        parallel=parallel,
        echo=typer.echo,
    )
    for label in runner.reclassify_existing():
        typer.echo(f"[{label}] earlier result reclassified as infrastructure; re-running")
    todo = len(runner.pending())
    typer.echo(
        f"{eval_id}: {len(tasks)} tasks x {runs} runs, {todo} attempt(s) to run "
        f"with {settings.llm.provider}/{settings.llm.model}"
    )
    with keep_awake() as awake:
        if awake:
            typer.echo("(system sleep is blocked while the eval runs)")
        summary = runner.run()

    metrics = compute_metrics(manifest, load_attempts(eval_dir, manifest_wall_limit(manifest)))
    md, _ = write_report(eval_dir, manifest, metrics)
    typer.echo(f"\nreport: {md}")
    typer.echo(
        f"resolve rate {metrics.resolved}/{metrics.valid_attempts} "
        f"of {metrics.planned_attempts} planned attempts"
    )
    if summary.stopped_reason:
        typer.echo(f"stopped early: {summary.stopped_reason}", err=True)
        typer.echo(
            f"resume with: repair-agent eval --resume {eval_id} --runs {runs}"
            + (f" --filter '{filter_}'" if filter_ != "*" else ""),
            err=True,
        )
        raise typer.Exit(code=3)


@app.command("report")
def report(
    eval_id: Annotated[str, typer.Argument(help="Eval id (directory name under runs/).")],
    append_results: Annotated[
        bool, typer.Option(help="Append this eval's headline row to RESULTS.md.")
    ] = False,
) -> None:
    """Recompute metrics from saved results and (re)write the report."""
    from repair_agent.eval.metrics import compute_metrics
    from repair_agent.eval.report import append_results_row, write_report
    from repair_agent.eval.runner import load_attempts, load_manifest, manifest_wall_limit

    eval_dir = get_settings().runs_dir / eval_id
    if not (eval_dir / "manifest.json").is_file():
        typer.echo(f"error: no eval {eval_id!r} found", err=True)
        raise typer.Exit(code=1)
    manifest = load_manifest(eval_dir)
    metrics = compute_metrics(manifest, load_attempts(eval_dir, manifest_wall_limit(manifest)))
    md, js = write_report(eval_dir, manifest, metrics)
    typer.echo(md.read_text(encoding="utf-8"))
    typer.echo(f"\nwrote {md} and {js}")
    if append_results:
        append_results_row(Path("RESULTS.md"), manifest, metrics)
        typer.echo("appended a row to RESULTS.md")


if __name__ == "__main__":
    app()
