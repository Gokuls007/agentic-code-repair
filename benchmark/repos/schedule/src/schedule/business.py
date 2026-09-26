from datetime import timedelta


def is_business_day(day, holidays=()):
    """Monday to Friday, excluding holidays."""
    return day.weekday() < 5 and day not in holidays


def add_business_days(day, n, holidays=()):
    """The date n business days after day. The starting day itself is not counted,
    so n=0 returns day unchanged."""
    if n < 0:
        raise ValueError("n must be non-negative")
    current = day
    remaining = n
    while remaining > 0:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            remaining -= 1
    return current
