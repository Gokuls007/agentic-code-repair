from decimal import Decimal

from inventory.models import Item
from inventory.pricing import discount, format_money, subtotal, total


def item(price, sku="X-1"):
    return Item(sku, "thing", Decimal(price))


def test_subtotal():
    lines = [(item("2.50"), 2), (item("1.00", "Y-1"), 3)]
    assert subtotal(lines) == Decimal("8.00")


def test_no_discount_below_threshold():
    assert discount(Decimal("99.99")) == Decimal("0.00")


def test_discount_at_threshold():
    assert discount(Decimal("100.00")) == Decimal("10.00")


def test_discount_above_threshold():
    assert discount(Decimal("250.00")) == Decimal("25.00")


def test_total_applies_discount_then_tax():
    assert total([(item("50.00"), 2)], 0.05) == Decimal("94.50")


def test_total_rounds_half_up():
    assert total([(item("2.50"), 1)], 0.07) == Decimal("2.68")


def test_format_money():
    assert format_money(Decimal("1234.5")) == "$1,234.50"
