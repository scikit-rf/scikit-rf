"""
skrf is an object-oriented approach to microwave engineering,
implemented in Python.
"""

__version__ = '2.0.0'
# Import all  module names for coherent reference of name-space
import os as _os

from . import (
    calibration,
    circuit,
    constants,
    data,
    frequency,
    instances,
    io,
    mathFunctions,
    media,
    network,
    networkSet,
    plotting,
    qfactor,
    taper,
    tlineFunctions,
    util,
    vectorFitting,
    vi,
)
from .calibration import calibrationSet, deembedding
from .frequency import Frequency
from .network import Network
from .plotting import stylely


def setup_plotting():
    plotting_environment = _os.environ.get('SKRF_PLOT_ENV', "pylab").lower()
    if plotting_environment == "pylab-skrf-style":
        stylely()
    # elif some different plotting environment
        # set that up


setup_plotting()
