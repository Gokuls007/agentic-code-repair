SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to"}
)


def title_case(text):
    """Title-case text. Small words stay lowercase, except as the first or last word."""
    words = text.split()
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        if 0 < i < len(words) - 1 and lower in SMALL_WORDS:
            out.append(lower)
        else:
            out.append(lower[:1].upper() + lower[1:])
    return " ".join(out)


def word_count(text):
    """Number of whitespace-separated words."""
    return len(text.split())
