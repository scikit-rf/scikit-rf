'''
skrf is an object-oriented approach to microwave engineering,
implemented in Python. 
'''
# Python 3 compatibility
from __future__ import absolute_import, print_function, division
from six.moves import xrange 

__version__ = '0.14.8'
## Import all  module names for coherent reference of name-space
#import io


from . import frequency
from . import network
from . import networkSet
from . import media

from . import calibration
# from . import plotting
from . import mathFunctions
from . import tlineFunctions
from . import taper
from . import constants
from . import util
from . import io
from . import instances


# Import contents into current namespace for ease of calling
from .frequency import *
from .network import *
from .networkSet import *
from .calibration import *
from .util import *
# from .plotting import  *
from .mathFunctions import *
from .tlineFunctions import *
from .io import * 
from .constants import * 
from .taper import * 
from .instances import * 

# Try to import vi, but if except if pyvisa not installed
try:
    import vi
    from vi import *
except(ImportError):
    pass

# try to import data but if it fails whatever. it fails if some pickles 
# dont unpickle. but its not important
try:
    from . import data
except:
    pass 



## built-in imports
from copy import deepcopy as copy


## Shorthand Names
F = Frequency
N = Network
NS = NetworkSet
lat = load_all_touchstones
# saf  = save_all_figs
saf = None
stylely = None


def setup_pylab():
    from . import plotting
    plotting.setup_matplotlib_plotting()

    global saf, stylely
    saf = plotting.save_all_figs
    stylely = plotting.stylely


def setup_plotting():
    plotting_environment = os.environ.get('SKRF_PLOT_ENV', "pylab").lower()

    if plotting_environment == "pylab":
        setup_pylab()
    elif plotting_environment == "pylab-skrf-style":
        setup_pylab()
        stylely()
    # elif some different plotting environment
        # set that up

setup_plotting()
