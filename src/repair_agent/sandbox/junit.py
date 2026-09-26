"""Parse pytest's JUnit XML into per-test outcomes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from enum import StrEnum

from pydantic import BaseModel, Field


class Outcome(StrEnum):
    """Result of one test case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TestReport(BaseModel):
    """Structured result of one pytest run inside the sandbox."""

    __test__ = False  # not a pytest test class

    outcomes: dict[str, Outcome] = Field(default_factory=dict)
    exit_code: int | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    oom: bool = False
    output: str = ""

    def count(self, outcome: Outcome) -> int:
        """Number of tests with ``outcome``."""
        return sum(1 for o in self.outcomes.values() if o == outcome)

    def ids_with(self, outcome: Outcome) -> list[str]:
        """Node ids with ``outcome``, sorted."""
        return sorted(k for k, v in self.outcomes.items() if v == outcome)


def node_id(classname: str, name: str, file: str | None = None) -> str:
    """Rebuild a pytest node id from JUnit ``classname``/``name`` attributes.

    pytest writes ``classname="tests.test_ops.TestX"`` and ``name="test_y[param]"``.
    Class segments start with an uppercase letter by pytest convention; the rest is
    the module path.
    """
    parts = classname.split(".") if classname else []
    classes: list[str] = []
    while parts and parts[-1][:1].isupper():
        classes.insert(0, parts.pop())
    module = file or ("/".join(parts) + ".py" if parts else "")
    return "::".join([module, *classes, name]) if module else name


def parse_junit(xml_text: str | bytes) -> dict[str, Outcome]:
    """Map node id -> outcome from a pytest JUnit XML report. Empty input gives {}."""
    if not xml_text or not xml_text.strip():
        return {}
    root = ET.fromstring(xml_text)
    outcomes: dict[str, Outcome] = {}
    for case in root.iter("testcase"):
        nid = node_id(case.get("classname", ""), case.get("name", ""), case.get("file"))
        if case.find("error") is not None:
            outcome = Outcome.ERROR
        elif case.find("failure") is not None:
            outcome = Outcome.FAILED
        elif case.find("skipped") is not None:
            outcome = Outcome.SKIPPED
        else:
            outcome = Outcome.PASSED
        # A test that fails and then errors in teardown appears twice; keep the worst.
        if outcomes.get(nid) not in (Outcome.ERROR, Outcome.FAILED):
            outcomes[nid] = outcome
    return outcomes
