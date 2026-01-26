"""
skrf is an object-oriented approach to microwave engineering,
implemented in Python.
"""

__version__ = '1.9.0'
## Import all  module names for coherent reference of name-space
#import io
import os as os_
import sys as sys_
from warnings import warn

from . import (
    calibration,
    circuit,
    constants,
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
)
from .calibration import calibrationSet, deembedding
from .circuit import Circuit
from .frequency import Frequency
from .network import Network
from .networkSet import NetworkSet

# Try to import vi, but if except if pyvisa not installed
try:
    from . import vi
except ImportError:
    pass

# try to import data but if it fails whatever. it fails if some pickles
# dont unpickle. but its not important
try:
    from . import data
except Exception:
    pass


# Defer imports for deprecated names and issue warnings
def __getattr__(name: str):
    if name not in ['__warningregistry__']:
        if 'vi' in sys_.modules:
            result = getattr(vi, os_.name, None)
            if result is not None:
                warn(f"skrf.{name} is deprecated. Please import {name} from skrf.vi instead.",
                     FutureWarning, stacklevel=2)
                return result
        for module in [
            vectorFitting,
            util,
            tlineFunctions,
            taper,
            qfactor,
            networkSet,
            network,
            mathFunctions,
            io,
            instances,
            frequency,
            constants,
            circuit,
            calibration,
            calibrationSet,
            deembedding,
        ]:
            result = getattr(module, name, None)
            if result is not None:
                warn(f"skrf.{name} is deprecated. Please import {name} from skrf.{
                    module.__name__.split('.')[-1]} instead.", FutureWarning, stacklevel=2)
                return result
        result = getattr(instances._instances, name, None)
        if result is not None:
            warn(f"skrf.{name} is deprecated. Please import {name} from skrf.instances instead.",
                 FutureWarning, stacklevel=2)
            return result
    raise AttributeError(f"module 'skrf' has no attribute '{name}'")

## built-in imports
# from copy import deepcopy as copy

## Shorthand Names
F = Frequency
N = Network
NS = NetworkSet
C = Circuit
# lat = load_all_touchstones
# saf  = save_all_figs
saf = None
stylely = None


def setup_pylab() -> bool:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not found while setting up plotting")
        return False


    global saf, stylely
    saf = plotting.save_all_figs
    stylely = plotting.stylely
    return True


def setup_plotting():
    plotting_environment = os_.environ.get('SKRF_PLOT_ENV', "pylab").lower()

    if plotting_environment == "pylab":
        setup_pylab()
    elif plotting_environment == "pylab-skrf-style":
        if setup_pylab():
            stylely()
    # elif some different plotting environment
        # set that up

plotting_available = setup_plotting()
