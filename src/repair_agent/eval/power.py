"""Keep the machine from sleeping while an eval runs.

A suspended host freezes in-flight requests for hours, so attempts time out without
the agent doing anything wrong (it happened in the first real Phase 4 eval). On Windows
we ask the OS to stay awake via ``SetThreadExecutionState``; elsewhere this is a no-op
(use ``caffeinate`` / ``systemd-inhibit`` if needed). The display may still turn off.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake() -> Iterator[bool]:
    """Prevent system sleep for the duration of the block. Yields True if supported."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    ok = bool(kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
    try:
        yield ok
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
