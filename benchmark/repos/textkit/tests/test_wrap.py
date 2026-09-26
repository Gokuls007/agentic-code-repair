import pytest
from textkit.wrap import truncate, wrap


def test_wrap_line_that_fits_exactly():
    assert wrap("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]


def test_wrap_breaks_long_text():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]


def test_wrap_keeps_overlong_word():
    assert wrap("supercalifragilistic word", 5) == ["supercalifragilistic", "word"]


def test_wrap_empty():
    assert wrap("", 5) == []


def test_wrap_invalid_width():
    with pytest.raises(ValueError):
        wrap("abc", 0)


def test_truncate_short_text_unchanged():
    assert truncate("hello", 8) == "hello"


def test_truncate_cuts_with_ellipsis():
    assert truncate("hello world", 8) == "hello w…"


def test_truncate_never_exceeds_width():
    for width in range(1, 12):
        assert len(truncate("hello world", width)) <= width


def test_truncate_tiny_width():
    assert truncate("hello", 1) == "…"
