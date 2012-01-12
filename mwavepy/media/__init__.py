#       media sub-module
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
#       MA 02110-1301, USA.from media import Media
'''
.. module:: mwavepy.media
========================================
media (:mod:`mwavepy.media`)
========================================

Provides :class:`media.Media` base-class and instances of :class:`media.Media` objects Class's for various 
transmission-line mediums.

Instances of the Media Class are objects which provide methods to
create network objects. See media for more detailed information.

Media basecalss
-------------------------

Transmission Line Classes
-------------------------
   
* :class:`~mwavepy.media.rectangularWaveguide.RectangularWaveguide` 
* :class:`~mwavepy.media.distributedCircuit.DistributedCircuite` 
* :class:`~mwavepy.media.cpw.CPW` 
* :class:`~mwavepy.media.freespace.Freespace` 


.. Hackk to generate docs for these classes, without displaying the 
    table
    .. autosummary::
	:toctree: generated/	
	:nosignatures:
	
	media.Media
	distributedCircuit.DistributedCircuit
	rectangularWaveguide.RectangularWaveguide
	cpw.CPW
	freespace.Freespace
	

'''

from media import Media
from distributedCircuit import DistributedCircuit
from freespace import Freespace
from cpw import CPW
from rectangularWaveguide import RectangularWaveguide
