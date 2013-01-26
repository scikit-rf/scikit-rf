
#       io.py
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
.. module:: skrf.io
========================================
io (:mod:`skrf.io`)
========================================


This Package provides functions and objects for input/output. 

The  general functions  :func:`~general.read` and :func:`~general.write` 
can be used to read and write [almost] any skrf object to disk, using the 
:mod:`pickle` module.

Reading and writing touchstone files is supported through the 
:class:`~touchstone.Touchstone` class, which can be more easily used 
through the Network constructor, :func:`~skrf.network.Network.__init__` 
 


.. automodule:: skrf.io.general
.. automodule:: skrf.io.touchstone
.. automodule:: skrf.io.csv

   
'''

import general
import csv
import touchstone

from general import * 
from csv import * 
from touchstone import * 
