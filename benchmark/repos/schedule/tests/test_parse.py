from datetime import date

import pytest
from schedule.parse import DateParser


def test_us_order():
    assert DateParser("MDY").parse("03/04/2024") == date(2024, 3, 4)


def test_european_order():
    assert DateParser("DMY").parse("03/04/2024") == date(2024, 4, 3)


def test_iso_dates_are_unambiguous():
    assert DateParser("DMY").parse("2024-12-25") == date(2024, 12, 25)


def test_default_order_is_us():
    assert DateParser().parse("01/02/2024") == date(2024, 1, 2)


def test_unsupported_order_rejected():
    with pytest.raises(ValueError):
        DateParser("YMD")
