import re
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
    assert v.validate_input(test_in) == expected


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
        v.validate_input(test_in)


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
    assert v.validate_input(test_in) == expected


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
        v.validate_input(test_in)


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
    assert v.validate_input(test_in) == expected


def test_enum_validator():
    class Foo(Enum):
        A = "A"
        B = "B"

    v = validators.EnumValidator(Foo)
    assert v.validate_input("A") == "A"
    assert v.validate_input("B") == "B"
    assert v.validate_input(Foo.A) == "A"
    assert v.validate_input(Foo.B) == "B"

    with pytest.raises(ValidationError):
        v.validate_input("C")

    assert v.validate_output('A') == Foo.A
    assert v.validate_output('B') == Foo.B
    with pytest.raises(ValidationError):
        v.validate_output("C")


def test_set_validator():
    v = validators.SetValidator([1, 2])
    assert v.validate_input(1) == 1
    assert v.validate_input(2) == 2
    with pytest.raises(ValidationError):
        v.validate_input(3)

    with pytest.raises(ValueError):
        validators.SetValidator([1, '2'])

def test_dict_validator():
    arg_str = "{a},{b}"
    resp_pat = r"(?P<a>\d),(?P<b>\d)"
    v = validators.DictValidator(arg_str, resp_pat)

    assert v.validate_input({"a":1, "b":2}) == "1,2"
    assert v.validate_output('1,2') == {"a":'1', "b":'2'}

    with pytest.raises(ValidationError):
        v.validate_input({"a":1})

    with pytest.raises(ValidationError):
        v.validate_output('1,2,3')

    v = validators.DictValidator(arg_str, re.compile(r"(?P<a>\d),(?P<b>\d)"))
    assert v.validate_input({"a":1, "b":2}) == "1,2"
