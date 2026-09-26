import json
from decimal import Decimal
from pathlib import Path

from inventory.models import Item, normalize_sku


class OutOfStock(Exception):
    """Raised when removing more units than are in stock."""


class Inventory:
    def __init__(self, items=()):
        self._items = {}
        for item in items:
            self.add(item)

    def add(self, item):
        if item.sku in self._items:
            raise ValueError(f"duplicate SKU {item.sku}")
        self._items[item.sku] = item

    def items(self):
        return list(self._items.values())

    def find(self, sku):
        """The item with this SKU (any capitalization), or None."""
        return self._items.get(normalize_sku(sku))

    def restock(self, sku, amount):
        self._require(sku).quantity += amount

    def remove_stock(self, sku, amount):
        item = self._require(sku)
        if amount > item.quantity:
            raise OutOfStock(f"only {item.quantity} of {item.sku} in stock")
        item.quantity -= amount

    def _require(self, sku):
        item = self._items.get(normalize_sku(sku))
        if item is None:
            raise KeyError(sku)
        return item


def save(inventory, path):
    data = [
        {"sku": i.sku, "name": i.name, "price": str(i.price), "quantity": i.quantity}
        for i in inventory.items()
    ]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load(path):
    """Load an inventory saved with save(). A missing file means an empty inventory."""
    path = Path(path)
    if not path.exists():
        return Inventory()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Inventory(Item(d["sku"], d["name"], Decimal(d["price"]), d["quantity"]) for d in data)
