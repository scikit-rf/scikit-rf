'''
#       workingBand.py
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
#import  mwavepy1 as mv1

from mwavepy1.frequency import Frequency
import mwavepy1 import createNetwork 

class WorkingBand(object):
	def __init__(self, frequency, tline):
		self.frequency = frequency 
		self.tline = tline
		self.f = self.frequency.axis

	def line(self,d):
		return createNetwork.delay(d=d, tline=self.tline, \
			frequency=self.f )
	def short(self):
		return createNetwork.short(self.f)
		
	def delay_short(self):
		return createNetwork.delay_short(d=d,tline=self.tline, \
			frequency = self.f)
	def 
