from decimal import Decimal

from inventory.models import Item
from inventory.report import low_stock, low_stock_report, stock_summary
from inventory.store import Inventory


def make():
    return Inventory(
        [
            Item("A-1", "apple", Decimal("1"), 2),
            Item("Z-1", "Zebra", Decimal("1"), 1),
            Item("M-1", "mango", Decimal("1"), 50),
        ]
    )


def test_low_stock_filters_by_threshold():
    assert [i.sku for i in low_stock(make(), 5)] == ["A-1", "Z-1"]


def test_low_stock_sorted_ignoring_case():
    assert [i.name for i in low_stock(make())] == ["apple", "Zebra"]


def test_low_stock_report_format():
    assert low_stock_report(make()) == "apple (A-1): 2\nZebra (Z-1): 1"


def test_threshold_is_inclusive():
    assert [i.sku for i in low_stock(make(), 1)] == ["Z-1"]


def test_stock_summary():
    assert stock_summary(make(), ["M-1", "Q-9"]) == {"M-1": 50, "Q-9": None}


def test_stock_summary_lowercase_sku():
    assert stock_summary(make(), ["m-1"]) == {"m-1": 50}
