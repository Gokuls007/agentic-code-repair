import re

WORD = re.compile(r"\w+")


def tokenize(text):
    """Words in text, case-folded, with their (start, end) character spans."""
    return [(m.group().casefold(), m.start(), m.end()) for m in WORD.finditer(text)]


def find_spans(text, term):
    """Spans of whole-word, case-insensitive occurrences of term in text."""
    needle = term.casefold()
    return [(start, end) for word, start, end in tokenize(text) if word == needle]
