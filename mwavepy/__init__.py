'''
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

## Import all  module names for coherent reference of name-space
import touchstone 	
import frequency
import transmissionLine	
import network
import workingBand
import calibration
import convenience
import plotting

# Import contents into current namespace for ease of calling 
from frequency import * 
from transmissionLine import * 
from network import * 
from workingBand import * 
from calibration import * 
from convenience import * 
from plotting import  * 

# Try to import virtualInstruments, but if except if pyvisa not installed
try:
	import virtualInstruments
	from virtualInstruments import *
except(ImportError):
	print '\nWARNING: pyvisa not installed, virtual instruments will not be available\n'
