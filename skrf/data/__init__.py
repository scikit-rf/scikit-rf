
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
from ..io.io import read

pwd = os.path.dirname(os.path.abspath(__file__))

ntwk1 = Network(os.path.join(pwd, 'ntwk1.s2p'))
line = Network(os.path.join(pwd, 'line.s2p'))
ring_slot = Network(os.path.join(pwd, 'ring slot.s2p'))
one_port_cal = read(os.path.join(pwd, 'one_port.cal'))
