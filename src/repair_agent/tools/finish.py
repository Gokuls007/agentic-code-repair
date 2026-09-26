"""finish: the agent declares it is done."""

from __future__ import annotations

from pydantic import Field

from repair_agent.tools.base import Tool, ToolArgs, ToolOutput


class Finish(Tool):
    """Signal completion. The agent loop stops when it sees ``metadata['finished']``."""

    name = "finish"
    description = (
        "Call once when the fix is complete (or you cannot make progress). "
        "summary: what was wrong, what you changed, and how you verified it."
    )

    class Args(ToolArgs):
        summary: str = Field(min_length=1, description="Short description of the fix.")

    def run(self, args: Args) -> ToolOutput:
        return ToolOutput(
            content="Finish recorded.", metadata={"finished": True, "summary": args.summary}
        )
