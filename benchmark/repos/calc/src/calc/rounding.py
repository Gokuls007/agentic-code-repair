from decimal import ROUND_HALF_UP, Decimal


def round_half_up(value, ndigits=0):
    """Round halves away from zero: 2.5 -> 3, -2.5 -> -3, 0.125 -> 0.13 (ndigits=2).

    Returns an int when ndigits is 0, otherwise a float.
    """
    quantum = Decimal(1).scaleb(-ndigits)
    result = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return float(result) if ndigits else int(result)
