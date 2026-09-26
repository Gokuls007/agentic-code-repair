from textkit.case import title_case, word_count


def test_title_case_small_words():
    assert title_case("the lord of the rings") == "The Lord of the Rings"


def test_title_case_last_word_capitalized():
    assert title_case("what are you looking at") == "What Are You Looking At"


def test_title_case_normalizes_mixed_case():
    assert title_case("hELLO wORLD") == "Hello World"


def test_title_case_single_small_word():
    assert title_case("the") == "The"


def test_word_count():
    assert word_count("one two  three") == 3


def test_word_count_mixed_whitespace():
    assert word_count("  a \t b\n") == 2


def test_word_count_empty():
    assert word_count("") == 0
