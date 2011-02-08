#	   loadCell.py
#
#	This file holds all VNA models
#
#	   Copyright 2011  alex arsenovic <arsenovic@virginia.edu>
#	   
#	   This program is free software; you can redistribute it and/or modify
#	   it under the terms of the GNU General Public License as published by
#	   the Free Software Foundation; either version 2 of the License, or
#	   (at your option) any later version.
#	   
#	   This program is distributed in the hope that it will be useful,
#	   but WITHOUT ANY WARRANTY; without even the implied warranty of
#	   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	   GNU General Public License for more details.
#	   
#	   You should have received a copy of the GNU General Public License
#	   along with this program; if not, write to the Free Software
#	   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#	   MA 02110-1301, USA.

'''
holds class's for load cells
'''

import ctypes as ctp
try:
	import d2xx:
except (ImportError):
	print ('WARNING: Module d2xx not found, Futek Loadcell VI not available.')
	
class Futek(object):
	def __init__(self):
		return 0
	def read(self):
		raise (NotImplementedError)
