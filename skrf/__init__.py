"""
skrf is an object-oriented approach to microwave engineering,
implemented in Python.
"""

__version__ = '1.10.0'
# Import all  module names for coherent reference of name-space
import os as _os
from typing import Any as _Any
from warnings import warn as _warn

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


# Defer imports for deprecated names and issue warnings
def __getattr__(name: str):
    result: _Any = None
    if name == 'N':
        from .network import Network as result
    elif name == 'F':
        from .frequency import Frequency as result
    elif name == 'NS':
        from .networkSet import NetworkSet as result
    elif name == 'C':
        from .circuit import Circuit as result
    elif name == 'saf':
        from .plotting import save_all_figs as result
    elif name == 'lat':
        from .io import load_all_touchstones as result
    if result is not None:
        _warn(f"Shorthand skrf.{name} is deprecated. Please use the full name instead.", FutureWarning, stacklevel=2)
        return result
    if name not in ['__warningregistry__']:
        for module in [
            vi,
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
                _warn(f"skrf.{name} is deprecated. Please import {name} from "
                     f"skrf.{module.__name__.split('.')[-1]} instead.", FutureWarning, stacklevel=2)
                return result
    raise AttributeError(f"module 'skrf' has no attribute '{name}'")


# Shorthand Names
stylely = None


def setup_pylab() -> bool:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not found while setting up plotting")
        return False

    global stylely
    stylely = plotting.stylely
    return True


def setup_plotting():
    plotting_environment = _os.environ.get('SKRF_PLOT_ENV', "pylab").lower()

    if plotting_environment == "pylab":
        setup_pylab()
    elif plotting_environment == "pylab-skrf-style":
        if setup_pylab():
            stylely()
    # elif some different plotting environment
        # set that up

plotting_available = setup_plotting()
