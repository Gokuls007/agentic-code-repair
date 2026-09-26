"""Text utilities: slugs, wrapping, casing, and simple search."""

from textkit.case import title_case, word_count
from textkit.highlight import highlight
from textkit.index import Index
from textkit.slug import slugify
from textkit.wrap import truncate, wrap

__all__ = ["Index", "highlight", "slugify", "title_case", "truncate", "word_count", "wrap"]
