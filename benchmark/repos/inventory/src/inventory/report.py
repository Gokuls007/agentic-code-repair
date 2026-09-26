from inventory.models import normalize_sku


def low_stock(inventory, threshold=5):
    """Items with quantity at or below threshold, sorted by name ignoring case."""
    items = [i for i in inventory.items() if i.quantity <= threshold]
    return sorted(items, key=lambda i: i.name.casefold())


def low_stock_report(inventory, threshold=5):
    return "\n".join(f"{i.name} ({i.sku}): {i.quantity}" for i in low_stock(inventory, threshold))


def stock_summary(inventory, skus):
    """Map each requested SKU (as given, any capitalization) to its quantity, or None."""
    by_sku = {i.sku: i.quantity for i in inventory.items()}
    return {s: by_sku.get(normalize_sku(s)) for s in skus}
