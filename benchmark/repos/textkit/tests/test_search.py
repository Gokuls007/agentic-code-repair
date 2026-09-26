from textkit.highlight import highlight
from textkit.index import Index
from textkit.tokens import find_spans, tokenize


def test_tokenize_spans():
    assert tokenize("hi there") == [("hi", 0, 2), ("there", 3, 8)]


def test_find_spans_whole_words_only():
    assert find_spans("cat concat cat", "cat") == [(0, 3), (11, 14)]


def test_highlight_basic():
    assert highlight("foo bar", "bar") == "foo [bar]"


def test_highlight_case_insensitive():
    assert highlight("Foo bar FOO", "foo") == "[Foo] bar [FOO]"


def test_highlight_no_match():
    assert highlight("foo bar", "baz") == "foo bar"


def test_index_lookup():
    idx = Index()
    idx.add("d1", "apple pie")
    idx.add("d2", "apple tart")
    assert idx.lookup("apple") == ["d1", "d2"]
    assert idx.lookup("pie") == ["d1"]


def test_index_case_insensitive():
    idx = Index()
    idx.add("d1", "Foo Bar")
    assert idx.lookup("foo") == ["d1"]
    assert idx.lookup("FOO") == ["d1"]


def test_index_unknown_term():
    assert Index().lookup("missing") == []
