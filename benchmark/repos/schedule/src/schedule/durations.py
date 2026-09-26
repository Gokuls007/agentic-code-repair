import re
from datetime import datetime

_PART = re.compile(r"(\d+)([dhms])")
_FORMAT = re.compile(r"(\d+[dhms])+")
_SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    """Seconds in a compact duration such as '45m', '2d' or '1h30m'."""
    text = text.strip().lower()
    if not _FORMAT.fullmatch(text):
        raise ValueError(f"invalid duration: {text!r}")
    return sum(int(amount) * _SECONDS[unit] for amount, unit in _PART.findall(text))


def time_until(target, now=None):
    """Time from now until target. `now` defaults to the current time in target's
    timezone (or naive local time for a naive target)."""
    if now is None:
        now = datetime.now(target.tzinfo)
    return target - now
