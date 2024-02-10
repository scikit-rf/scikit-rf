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


#from parametricStandard import *
from . import calibration, calibrationSet, deembedding
from .calibration import *
from .calibrationSet import *
from .deembedding import *
