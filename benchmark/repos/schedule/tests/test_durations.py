from datetime import datetime, timedelta, timezone

import pytest
from schedule.durations import parse_duration, time_until


def test_single_unit():
    assert parse_duration("45m") == 2700
    assert parse_duration("2d") == 172800


def test_combined_units():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("1d2h3m4s") == 93784


def test_case_and_whitespace():
    assert parse_duration(" 2H ") == 7200


@pytest.mark.parametrize("bad", ["", "5", "h1", "1x", "1h 30m"])
def test_invalid_durations(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_time_until_with_explicit_now():
    now = datetime(2024, 1, 1, 12)
    assert time_until(datetime(2024, 1, 1, 13), now) == timedelta(hours=1)


def test_time_until_naive_target():
    target = datetime.now() + timedelta(hours=1)
    assert timedelta(minutes=59) < time_until(target) <= timedelta(hours=1)


def test_time_until_aware_target():
    tz = timezone(timedelta(hours=-5))
    target = datetime.now(tz) + timedelta(hours=2)
    assert timedelta(hours=1, minutes=59) < time_until(target) <= timedelta(hours=2)
