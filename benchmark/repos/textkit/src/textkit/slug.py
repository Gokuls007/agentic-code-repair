import re
import unicodedata


def slugify(text, sep="-"):
    """URL slug: accents transliterated to ASCII, lowercase, and every run of
    non-alphanumeric characters replaced by a single separator."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    words = re.split(r"[^a-z0-9]+", ascii_text.lower())
    return sep.join(w for w in words if w)
