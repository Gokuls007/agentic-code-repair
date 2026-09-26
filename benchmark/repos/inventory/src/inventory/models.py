from dataclasses import dataclass
from decimal import Decimal


def normalize_sku(sku):
    """SKUs are case-insensitive; the canonical form is stripped and upper-case."""
    return sku.strip().upper()


@dataclass
class Item:
    sku: str
    name: str
    price: Decimal
    quantity: int = 0

    def __post_init__(self):
        self.sku = normalize_sku(self.sku)
        self.price = Decimal(str(self.price))
        if self.quantity < 0:
            raise ValueError("quantity cannot be negative")
