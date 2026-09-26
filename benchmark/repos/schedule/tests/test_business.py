from datetime import date

import pytest
from schedule.business import add_business_days, is_business_day

FRIDAY = date(2024, 3, 1)


def test_weekdays_are_business_days():
    assert all(is_business_day(date(2024, 3, d)) for d in (4, 5, 6, 7, 8))


def test_weekend_is_not_a_business_day():
    assert not is_business_day(date(2024, 3, 2))
    assert not is_business_day(date(2024, 3, 3))


def test_holiday_is_not_a_business_day():
    assert not is_business_day(date(2024, 3, 4), holidays={date(2024, 3, 4)})


def test_add_one_business_day_from_friday():
    assert add_business_days(FRIDAY, 1) == date(2024, 3, 4)


def test_add_zero_business_days_is_same_day():
    assert add_business_days(FRIDAY, 0) == FRIDAY


def test_add_business_days_skips_holidays():
    assert add_business_days(FRIDAY, 1, holidays={date(2024, 3, 4)}) == date(2024, 3, 5)


def test_add_five_business_days():
    assert add_business_days(date(2024, 3, 4), 5) == date(2024, 3, 11)


def test_negative_days_rejected():
    with pytest.raises(ValueError):
        add_business_days(FRIDAY, -1)
