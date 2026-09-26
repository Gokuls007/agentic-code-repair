from calc.ops import add, clamp, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(2, 3) == -1


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2
