#       calibration sub-module
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.from media import Media
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
import parametricStandard
import calibrationFunctions

from parametricStandard import *
from calibration import Calibration
from calibrationFunctions import *
