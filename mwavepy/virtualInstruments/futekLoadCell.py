#       futekLoadCell.py
#       
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
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

import subprocess as sbp
class Futek_USB210(object):
	'''
	'''
	def __init__(self):
		self.process = sbp.Popen(['futek_usb210.exe'],stdin=sbp.PIPE,\
			stdout=sbp.PIPE)
		
	@property
	def data(self):
		stdout, stderr = self.process.communicate('1')
		if stderr is None:
			return (float(stdout))
		else:
			print ('ERROR from futek:%s'%stderr)
			return (float(stdout))
	def close(self):
		self.process.terminate()
	
