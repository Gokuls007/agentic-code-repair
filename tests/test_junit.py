from __future__ import annotations

from repair_agent.sandbox.junit import Outcome, TestReport, node_id, parse_junit

XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5">
  <testcase classname="tests.test_ops" file="tests/test_ops.py" name="test_add" time="0.001"/>
  <testcase classname="tests.test_ops" file="tests/test_ops.py" name="test_sub">
    <failure message="assert -1 == 1">boom</failure>
  </testcase>
  <testcase classname="tests.test_ops.TestClamp" file="tests/test_ops.py" name="test_low[0-1]"/>
  <testcase classname="tests.test_ops" file="tests/test_ops.py" name="test_skip">
    <skipped message="later"/>
  </testcase>
  <testcase classname="" name="tests.test_broken"><error message="collection failure"/></testcase>
</testsuite></testsuites>"""


def test_parse_junit_outcomes() -> None:
    assert parse_junit(XML) == {
        "tests/test_ops.py::test_add": Outcome.PASSED,
        "tests/test_ops.py::test_sub": Outcome.FAILED,
        "tests/test_ops.py::TestClamp::test_low[0-1]": Outcome.PASSED,
        "tests/test_ops.py::test_skip": Outcome.SKIPPED,
        "tests.test_broken": Outcome.ERROR,
    }


def test_parse_junit_empty() -> None:
    assert parse_junit("") == {}
    assert parse_junit(b"   ") == {}


def test_failure_then_teardown_error_keeps_worst() -> None:
    xml = """<testsuites><testsuite>
      <testcase classname="t" file="t.py" name="test_a"><failure/></testcase>
      <testcase classname="t" file="t.py" name="test_a"/>
    </testsuite></testsuites>"""
    assert parse_junit(xml) == {"t.py::test_a": Outcome.FAILED}


def test_node_id_without_file_attribute() -> None:
    assert node_id("tests.sub.test_x.TestA", "test_b") == "tests/sub/test_x.py::TestA::test_b"


def test_report_helpers() -> None:
    report = TestReport(outcomes=parse_junit(XML))
    assert report.count(Outcome.PASSED) == 2
    assert report.ids_with(Outcome.FAILED) == ["tests/test_ops.py::test_sub"]
