"""Regression tests for the module-level ``__getattr__`` in ``skrf/__init__.py``.

Background
----------
``skrf/__init__.py`` defines a module-level ``__getattr__`` that lets callers
write ``skrf.X`` for attributes that actually live in sub-modules, while
emitting a ``FutureWarning`` that the shorthand is deprecated. The original
implementation walked every sub-module looking for ``X`` and warned
whenever it was found — including for symbols that the sub-module had
merely re-exported from ``numpy`` or ``scipy`` (``skrf.calibration`` does
``from numpy import angle``, so ``skrf.angle`` resolves to ``numpy.angle``).

Issue #1388 reported the resulting

    >>> import skrf as rf
    >>> rf.angle(0)
    FutureWarning: skrf.angle is deprecated. Please import angle from
    skrf.calibration instead.

as misleading: ``np.angle`` is the canonical home of that function, not
``skrf.calibration``. Commit ``b1d4969`` ("avoid throwing warnings for
numpy/scipy imported functions") short-circuits the warning whenever the
resolved attribute's ``__module__`` starts with ``numpy`` or ``scipy``.
The tests below pin that behaviour so the suppression cannot silently
regress under future refactors of ``__getattr__``.
"""

from __future__ import annotations

import warnings

import numpy as np

import skrf


def _collect_warnings(callable_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = callable_()
    return result, caught


def test_skrf_angle_does_not_warn():
    """Regression for #1388.

    ``skrf.angle`` resolves to ``numpy.angle`` (re-exported via
    ``skrf.calibration``). Accessing it through the module-level
    ``__getattr__`` must not emit a ``FutureWarning``.
    """
    result, caught = _collect_warnings(lambda: skrf.angle)

    assert result is np.angle, (
        "skrf.angle should resolve to numpy.angle; got "
        f"{result!r} from module {getattr(result, '__module__', '?')!r}"
    )
    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future_warnings, (
        "skrf.angle resolves to numpy.angle and must not raise a "
        "FutureWarning; got: "
        + ", ".join(str(w.message) for w in future_warnings)
    )


def test_skrf_numpy_function_reexport_does_not_warn():
    """Companion to ``test_skrf_angle_does_not_warn``: any numpy function
    that some sub-module happens to re-export should also pass through
    silently. Using ``exp`` as a second probe so a single-symbol
    short-circuit cannot accidentally pass the suite."""
    if getattr(skrf, "exp", None) is not np.exp:
        # If a future skrf version stops re-exporting ``exp`` this assertion
        # is no longer meaningful; the angle case above still pins #1388.
        return

    result, caught = _collect_warnings(lambda: skrf.exp)

    assert result is np.exp
    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future_warnings, (
        "skrf.exp resolves to numpy.exp and must not raise a "
        "FutureWarning; got: "
        + ", ".join(str(w.message) for w in future_warnings)
    )


def test_skrf_native_shorthand_still_warns_as_control():
    """Positive control: the numpy/scipy suppression must not be
    over-broad. ``skrf.N`` is a skrf-native one-letter shorthand for
    ``skrf.network.Network`` and is explicitly meant to be deprecated, so
    accessing it must still emit the shorthand FutureWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = skrf.N

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert result is skrf.network.Network
    assert any(
        "Shorthand skrf.N is deprecated" in str(w.message)
        for w in future_warnings
    ), (
        "Expected the shorthand-deprecation FutureWarning for skrf.N; "
        "the numpy/scipy suppression must not apply to skrf-native names."
    )
