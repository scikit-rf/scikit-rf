
#       __init__.py
#
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#
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
#       MA 02110-1301, USA.
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

import touchstone
import frequency
import network
import networkSet
import convenience
import plotting
import mathFunctions
import tlineFunctions


# Import contents into current namespace for ease of calling
from frequency import *
from network import *
from networkSet import *
from calibration import *
from convenience import *
from plotting import  *
from mathFunctions import *
from tlineFunctions import *

# Try to import virtualInstruments, but if except if pyvisa not installed
try:
    import virtualInstruments
    from virtualInstruments import *
except(ImportError):
    print '\nWARNING: pyvisa not installed, virtual instruments will not be available\n'



## built-in imports
from copy import deepcopy as copy
