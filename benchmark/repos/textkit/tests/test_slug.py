from textkit.slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_runs_collapse():
    assert slugify("Hello,  World!") == "hello-world"


def test_accents_transliterated():
    assert slugify("Café Déjà Vu") == "cafe-deja-vu"


def test_custom_separator():
    assert slugify("a b c", sep="_") == "a_b_c"


def test_leading_and_trailing_punctuation():
    assert slugify("  --Hi--  ") == "hi"


def test_only_punctuation():
    assert slugify("!!!") == ""
