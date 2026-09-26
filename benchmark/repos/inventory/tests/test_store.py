from decimal import Decimal

import pytest
from inventory.models import Item
from inventory.store import Inventory, OutOfStock, load, save


def make():
    return Inventory(
        [
            Item("abc-1", "Widget", Decimal("2.50"), 10),
            Item("XYZ-9", "Gadget", Decimal("9.99"), 0),
        ]
    )


def test_skus_are_normalized():
    assert [i.sku for i in make().items()] == ["ABC-1", "XYZ-9"]


def test_find_exact():
    assert make().find("ABC-1").name == "Widget"


def test_find_is_case_insensitive():
    assert make().find("abc-1").name == "Widget"


def test_find_unknown():
    assert make().find("nope") is None


def test_restock_lowercase_sku():
    inv = make()
    inv.restock("xyz-9", 5)
    assert inv.find("XYZ-9").quantity == 5


def test_remove_stock():
    inv = make()
    inv.remove_stock("ABC-1", 4)
    assert inv.find("ABC-1").quantity == 6


def test_remove_too_much_raises():
    inv = make()
    with pytest.raises(OutOfStock):
        inv.remove_stock("ABC-1", 13)
    assert inv.find("ABC-1").quantity == 10


def test_duplicate_sku_rejected():
    with pytest.raises(ValueError):
        Inventory([Item("a-1", "One", Decimal("1")), Item("A-1", "Two", Decimal("1"))])


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "store.json"
    save(make(), path)
    loaded = load(path)
    assert [(i.sku, i.name, i.price, i.quantity) for i in loaded.items()] == [
        ("ABC-1", "Widget", Decimal("2.50"), 10),
        ("XYZ-9", "Gadget", Decimal("9.99"), 0),
    ]


def test_load_missing_file_gives_empty_inventory(tmp_path):
    assert load(tmp_path / "missing.json").items() == []
