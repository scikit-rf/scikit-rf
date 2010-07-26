'''
#       constants.py
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
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

from transmissionLine import RectangularWaveguideTE10
from frequency import Frequency
from workingBand import WorkingBand

from scipy.constants import mil

f_wr1p5 = Frequency(500,750,201, 'ghz')
#f_wr3 = Frequency(500,750,201, 'ghz')

wg_wr1p5 = RectangularWaveguideTE10(1.5*10*mil)
wg_wr3 = RectangularWaveguideTE10(3*10*mil)


wb_wr1p5 = WorkingBand(frequency = f_wr1p5, tline = wg_wr1p5)
