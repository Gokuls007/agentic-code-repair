import math


def mean(xs):
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def median(xs):
    if not xs:
        raise ValueError("median of empty sequence")
    ordered = sorted(xs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def variance(xs, sample=False):
    """Population variance by default, or sample variance (divides by n - 1).

    Raises ValueError for an empty sequence, and for fewer than two values when
    sample=True.
    """
    n = len(xs)
    if n == 0:
        raise ValueError("variance of empty sequence")
    m = mean(xs)
    ss = sum((x - m) ** 2 for x in xs)
    if sample:
        if n < 2:
            raise ValueError("sample variance needs at least two values")
        return ss / (n - 1)
    return ss / n


def percentile(xs, p):
    """Linearly interpolated percentile of xs, for p between 0 and 100."""
    if not xs:
        raise ValueError("percentile of empty sequence")
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(xs)
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
