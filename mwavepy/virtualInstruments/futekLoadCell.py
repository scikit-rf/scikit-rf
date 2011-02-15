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

class Futek_USB210(object):
	'''
	'''
	def __init__(self):
		dir = os.path.dirname(__file__)
		self.process = sbp.Popen([dir+'/test.out'],stdin=sbp.PIPE,\
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
	def __init__(self,ax=None, window_length = -1, **kwargs):
		self.futek = Futek_USB210(**kwargs)
		if ax is None:
			ax = plb.gca()
		self.ax = ax
		self.window_length = window_length
		self.xdata, self.ydata = [],[]

		self.update_data()
		self.line = ax.plot(self.xdata, self.ydata )[0]
		self.update_axis_scale()

	def get_data_and_plot(self):
		self.update_data()
		self.update_line()
		self.update_axis_scale()
		self.ax.figure.canvas.draw()
		
	def update_data(self):
		self.ydata.append( self.futek.data)
		self.xdata.append(len(self.xdata))
		#crop data 
		if self.window_length >0 and \
			len(self.xdata) > self.window_length:
			self.xdata = self.xdata [1:]
			self.ydata = self.ydata [1:]
	def update_line(self):
		self.line.set_data(self.xdata, self.ydata)
	def update_axis_scale(self):
		self.ax.axis([\
			self.line.get_xdata().min(),\
			self.line.get_xdata().max(),\
			self.line.get_ydata().min(),\
			self.line.get_ydata().max(),\
			])
	
