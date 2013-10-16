
'''
skrf is an object-oriented approach to microwave engineering,
implemented in the Python programming language. It provides a set of
objects and features which can be used to build powerful solutions to
specific problems. skrf's abilities are; touchstone file manipulation,
calibration, VNA data acquisition, circuit design and much more.

This is the main module file for skrf. it simply imports classes and
methods. It does this in two ways; import all into the current namespace,
and import modules themselves for coherent  structured referencing
'''

## Import all  module names for coherent reference of name-space
import media
import calibration
import io

import frequency
import network
import networkSet
import plotting
import mathFunctions
import tlineFunctions
import constants
import util

try:
    import data
except:
    print 'warning: data module didnt load. dont worry about it.'
    pass 
# Import contents into current namespace for ease of calling
from frequency import *
from network import *
from networkSet import *
from calibration import *
from util import *
from plotting import  *
from mathFunctions import *
from tlineFunctions import *
from io import * 
from constants import * 

# Try to import vi, but if except if pyvisa not installed
try:
    import vi
    from vi import *
except(ImportError):
    print '\nWARNING: pyvisa not installed, virtual instruments will not be available\n'



## built-in imports
from copy import deepcopy as copy


## Shorthand Names
F = Frequency
M = Media
N = Network
NS = NetworkSet
C = Calibration
lat = load_all_touchstones
saf  = save_all_figs

