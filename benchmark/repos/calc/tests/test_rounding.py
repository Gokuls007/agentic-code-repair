from calc.rounding import round_half_up


def test_half_up_to_integer():
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4


def test_half_up_negative():
    assert round_half_up(-2.5) == -3


def test_half_up_with_digits():
    assert round_half_up(0.125, 2) == 0.13
    assert round_half_up(2.675, 2) == 2.68


def test_no_rounding_needed():
    assert round_half_up(1.2, 1) == 1.2
    assert round_half_up(4) == 4


def test_returns_int_without_digits():
    assert isinstance(round_half_up(2.4), int)
