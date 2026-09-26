from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
DISCOUNT_THRESHOLD = Decimal("100.00")
DISCOUNT_RATE = Decimal("0.10")


def subtotal(lines):
    """Sum of price * quantity over (item, quantity) order lines."""
    return sum((item.price * qty for item, qty in lines), Decimal("0"))


def discount(amount):
    """10% off orders of $100.00 or more."""
    if amount >= DISCOUNT_THRESHOLD:
        return (amount * DISCOUNT_RATE).quantize(CENT, rounding=ROUND_HALF_UP)
    return Decimal("0.00")


def total(lines, tax_rate):
    """Subtotal minus discount, plus tax, rounded half-up to the cent once at the end."""
    base = subtotal(lines)
    discounted = base - discount(base)
    taxed = discounted * (1 + Decimal(str(tax_rate)))
    return taxed.quantize(CENT, rounding=ROUND_HALF_UP)


def format_money(amount):
    return f"${amount:,.2f}"
