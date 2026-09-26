from datetime import date, datetime, timedelta, timezone

from schedule.recur import monthly, next_occurrence


def test_monthly_keeps_day_of_month():
    assert monthly(date(2024, 1, 15), 3) == [
        date(2024, 1, 15),
        date(2024, 2, 15),
        date(2024, 3, 15),
    ]


def test_monthly_from_month_end():
    assert monthly(date(2024, 1, 31), 4) == [
        date(2024, 1, 31),
        date(2024, 2, 29),
        date(2024, 3, 31),
        date(2024, 4, 30),
    ]


def test_monthly_across_year_boundary():
    assert monthly(date(2023, 12, 10), 2) == [date(2023, 12, 10), date(2024, 1, 10)]


def test_monthly_zero_count():
    assert monthly(date(2024, 1, 1), 0) == []


def test_next_occurrence():
    assert next_occurrence(datetime(2024, 1, 31, 9), datetime(2024, 3, 1)) == datetime(
        2024, 3, 31, 9
    )


def test_next_occurrence_is_strictly_after():
    start = datetime(2024, 1, 15, 9)
    assert next_occurrence(start, datetime(2024, 2, 15, 9)) == datetime(2024, 3, 15, 9)


def test_next_occurrence_keeps_timezone():
    tz = timezone(timedelta(hours=1))
    start = datetime(2024, 1, 15, 9, tzinfo=tz)
    result = next_occurrence(start, datetime(2024, 2, 1, tzinfo=tz))
    assert result == datetime(2024, 2, 15, 9, tzinfo=tz)
    assert result.tzinfo is not None
