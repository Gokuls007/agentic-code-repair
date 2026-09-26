from dateutil import parser


class DateParser:
    """Parse human-entered dates in a configured field order: 'MDY' (US) or 'DMY'."""

    def __init__(self, order="MDY"):
        if order not in ("MDY", "DMY"):
            raise ValueError(f"unsupported order {order!r}")
        self.order = order

    def parse(self, text):
        return parser.parse(text, dayfirst=self.order == "DMY").date()
