"""
.. module:: skrf.calibration
========================================
calibration (:mod:`skrf.calibration`)
========================================


This Package provides functionality for performing and testing
calibration algorithms. Most functionality is in the :mod:`calibration`
module.

.. automodule:: skrf.calibration.calibration

"""
from warnings import warn as _warn

#from parametricStandard import *
from . import calibration, calibrationSet, deembedding
from .calibration import *


def __getattr__(name: str):
    if name not in ['__warningregistry__']:
        for module in [
            calibrationSet,
            deembedding,
        ]:
            result = getattr(module, name, None)
            if result is not None:
                _warn(f"skrf.calibration.{name} is deprecated. Please import {name} from "
                     f"skrf.calibration.{module.__name__.split('.')[-1]} instead.", FutureWarning, stacklevel=2)
                return result
    raise AttributeError(f"module 'skrf.calibration' has no attribute '{name}'")
