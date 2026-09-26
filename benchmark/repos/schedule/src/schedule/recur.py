from dateutil.relativedelta import relativedelta


def monthly(start, count):
    """count dates one calendar month apart, keeping start's day of month where the month
    allows it (Jan 31 -> Feb 29 -> Mar 31 -> Apr 30)."""
    return [start + relativedelta(months=i) for i in range(count)]


def next_occurrence(start, after, months=1):
    """First datetime in the series start, start + months, start + 2*months, ... that is
    strictly after `after`. The result keeps start's timezone."""
    i = 0
    while True:
        candidate = start + relativedelta(months=i * months)
        if candidate > after:
            return candidate
        i += 1
