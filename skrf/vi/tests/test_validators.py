from enum import Enum

import pytest

from skrf.vi import validators
from skrf.vi.validators import ValidationError


@pytest.mark.parametrize(
    "bounds,test_in,expected",
    [
        ((None, None), 0, 0),
        ((None, None), 0.0, 0),
        ((None, None), "0", 0),
        ((None, None), "-1", -1),
        ((None, None), "1", 1),
        ((None, None), "100000", 100000),
        ((-1, None), "-1", -1),
        ((None, 1), "0", 0),
        ((-1, 1), 1, 1),
        ((0, 100), "1", 1),
    ],
)
def test_int_validator(bounds, test_in, expected):
    v = validators.IntValidator(*bounds)
    assert v.validate(test_in) == expected


@pytest.mark.parametrize(
    "bounds,test_in",
    [
        ((-1, 1), 2),
        ((-1, 1), -2.0),
        ((-1, 1), "-2"),
        ((-1, None), -2),
        ((None, 1), 2),
        ((None, 1), 2),
    ],
)
def test_int_validator_out_of_bounds(bounds, test_in):
    v = validators.IntValidator(*bounds)
    with pytest.raises(ValidationError):
        v.validate(test_in)


@pytest.mark.parametrize(
    "bounds,test_in,expected",
    [
        ((None, None), 0.0, 0.0),
        ((None, None), 0, 0.0),
        ((None, None), "0", 0.0),
        ((None, None), "0.", 0.0),
        ((None, None), "-1", -1.0),
        ((None, None), "1.0", 1.0),
        ((None, None), "2.5", 2.5),
        ((None, None), "-2.5", -2.5),
        ((-1, None), "0", 0),
        ((None, 1), "0", 0),
        ((-0.5, 0.5), "0.25", 0.25),
    ],
)
def test_float_validator(bounds, test_in, expected):
    v = validators.FloatValidator(*bounds)
    assert v.validate(test_in) == expected


@pytest.mark.parametrize(
    "bounds,test_in",
    [
        ((-1, 1), 2),
        ((-1, 1), -2),
        ((-1, 1), "-2.0"),
        ((-1, None), -2),
        ((None, 1), 2),
    ],
)
def test_float_validator_out_of_bounds(bounds, test_in):
    v = validators.FloatValidator(*bounds)
    with pytest.raises(ValidationError):
        v.validate(test_in)


@pytest.mark.parametrize(
    "test_in,expected",
    [
        ("100", 100),
        ("1hz", 1),
        ("1hZ", 1),
        ("1Hz", 1),
        ("1HZ", 1),
        ("1 Hz", 1),
        ("1 kHz", 1000),
        ("1 MHz", 1_000_000),
        ("1 GHz", 1_000_000_000),
        ("1.5 GHz", 1_500_000_000),
        ("1 kHz", 1000),
    ],
)
def test_freq_validator(test_in, expected):
    v = validators.FreqValidator()
    assert v.validate(test_in) == expected


def test_enum_validator():
    class Foo(Enum):
        A = "A"
        B = "B"

    v = validators.EnumValidator(Foo)
    assert v.validate("A") == "A"
    assert v.validate("B") == "B"
    assert v.validate(Foo.A) == "A"
    assert v.validate(Foo.B) == "B"

    with pytest.raises(ValidationError):
        v.validate("C")


def test_set_validator():
    v = validators.SetValidator([1, 2])
    assert v.validate(1) == 1
    assert v.validate(2) == 2
    with pytest.raises(ValidationError):
        v.validate(3)
