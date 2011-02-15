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
import os
import pylab as plb
from matplotlib.lines import Line2D
class Futek_USB210(object):
	'''
	'''
	def __init__(self):
		dir = os.path.dirname(__file__)
		self.process = sbp.Popen([dir+'/std_in_out_win32'],stdin=sbp.PIPE,\
			stdout=sbp.PIPE)
		
	@property
	def data(self):
		self.write()
		return float(self.read())
	
	def write(self,data='44\n'):
		self.process.stdin.write(data)
	
	def read(self):
		return (self.process.stdout.readline())
		
	def close(self):
		self.process.terminate()
	

class FutekMonitor(object):
	def __init__(self,ax=None):
		self.futek = Futek_USB210()
		if ax is None:
			ax = plb.gca()
		self.ax = ax
		self.xdata, self.ydata = [],[]

		poll_data()
		self.line = Line2D(self.xdata, self.ydata)
		

	def poll_data(self):
		self.ydata.append( self.futek.data)
		self.xdata.append(len(self.xdata))

	def update_axis_scale(self):
		self.ax.axis([\
			self.line.get_xdata().min(),\
			self.line.get_xdata().max(),\
			self.line.get_ydata().min(),\
			self.line.get_ydata().max(),\
			 l.get_xdata().max(), l.get_ydata().min(), l.get_ydata().max()])
