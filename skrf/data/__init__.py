
#       data.py
#
#
#       Copyright 2012 alex arsenovic <arsenovic@virginia.edu>
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
.. module:: skrf.data
========================================
io (:mod:`skrf.data`)
========================================


This Package provides data to be used in examples and testcases

Modules
----------
.. toctree::
   :maxdepth: 1

   
'''
import os 

from  ..network import Network
from ..io.general import read

pwd = os.path.dirname(os.path.abspath(__file__))

ntwk1 = Network(os.path.join(pwd, 'ntwk1.s2p'))
line = Network(os.path.join(pwd, 'line.s2p'))
ring_slot = Network(os.path.join(pwd, 'ring slot.s2p'))
ring_slot_meas = Network(os.path.join(pwd, 'ring slot measured.s1p'))
wr2p2_line = Network(os.path.join(pwd, 'wr2p2,line.ntwk'))
wr2p2_line1 = Network(os.path.join(pwd, 'wr2p2,line1.ntwk'))
wr2p2_delayshort = Network(os.path.join(pwd, 'wr2p2,delayshort.ntwk'))
wr2p2_short = Network(os.path.join(pwd, 'wr2p2,short.ntwk'))
wr1p5_line = Network(os.path.join(pwd, 'wr1p5,line.ntwk'))
wr1p5_short = Network(os.path.join(pwd, 'wr1p5,short.ntwk'))

one_port_cal = read(os.path.join(pwd, 'one_port.cal'))
