
#       __init__.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen 
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
**mwavepy** is a compilation of functions and class's for microwave/RF
engineering written in python. It is useful for things such as
touchstone file manipulation, calibration, data analysis, data
acquisition, and plotting. mwavepy can be used interactively through
the python interpreter, or in scripts. 

This is the main module file for mwavepy. it simply imports classes and
methods. It does this in two ways; import all into the current namespace,
and import modules themselves for coherent  structured referencing
'''

## Import all  module names for coherent reference of name-space
import touchstone 	
import frequency
import transmissionLine	
import network
import workingBand
import discontinuities
import calibration
import convenience
import plotting
import mathFunctions




# Import contents into current namespace for ease of calling 
from frequency import * 
from transmissionLine import * 
from network import * 
from workingBand import * 
from calibration import * 
from convenience import * 
from plotting import  * 
from mathFunctions import *

# Try to import virtualInstruments, but if except if pyvisa not installed
try:
	import virtualInstruments
	from virtualInstruments import *
except(ImportError):
	print '\nWARNING: pyvisa not installed, virtual instruments will not be available\n'



## built-in imports
from copy import deepcopy as copy
