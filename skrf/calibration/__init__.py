
'''
.. module:: skrf.calibration
========================================
calibration (:mod:`skrf.calibration`)
========================================


This Package provides a high-level class representing a
calibration instance, as well as calibration algorithms and supporting
functions.

Both one and two port calibrations are supported. These calibration
algorithms allow for redundant measurements, by using a simple least
squares estimator to solve for the embedding network.

Modules
----------
.. toctree::
   :maxdepth: 1

   calibration
   calibrationAlgorithms
   calibrationFunctions


Classes
-------------------------
.. currentmodule:: skrf.calibration.calibration
.. autosummary::
        :toctree: generated/

        Calibration

'''

import calibration
import calibrationFunctions
import parametricStandard
import calibrationSet

from parametricStandard import *
from calibration import *
from calibrationFunctions import *
from calibrationSet import *
