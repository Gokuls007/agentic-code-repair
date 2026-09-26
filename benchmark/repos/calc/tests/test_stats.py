import pytest
from calc.stats import mean, median, percentile, variance


def test_mean_basic():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_single():
    assert mean([5]) == 5.0


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_median_two_values():
    assert median([1, 3]) == 2.0


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


def test_variance_population():
    assert variance([2, 4, 4, 4, 5, 5, 7, 9]) == 4.0


def test_variance_sample():
    assert variance([1, 2, 3, 4], sample=True) == pytest.approx(5 / 3)


def test_variance_single_value_population():
    assert variance([7]) == 0.0


def test_sample_variance_needs_two_values():
    with pytest.raises(ValueError):
        variance([7], sample=True)


def test_variance_empty_raises():
    with pytest.raises(ValueError):
        variance([])


def test_percentile_median():
    assert percentile([10, 20, 30, 40], 50) == 25


def test_percentile_bounds():
    assert percentile([10, 20, 30, 40], 0) == 10
    assert percentile([10, 20, 30, 40], 100) == 40


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 25) == 2


def test_percentile_invalid_p():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([1, 2], -1)
