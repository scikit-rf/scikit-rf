
#
#       convenience.py
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

.. currentmodule:: skrf.convenience
========================================
convenience (:mod:`skrf.convenience`)
========================================

Holds pre-initialized  objects's and functions that are general
conveniences.


Functions
------------
.. autosummary::
   :toctree: generated/

   save_all_figs
   add_markers_to_lines
   legend_off
   now_string
   find_nearest
   find_nearest_index


Pre-initialized Objects
--------------------------

:class:`~skrf.frequency.Frequency` Objects
==============================================
These are predefined :class:`~skrf.frequency.Frequency` objects
that correspond to standard waveguide bands. This information is taken
from the VDI Application Note 1002 [#]_ . The naming convenction is
f_wr# where '#' is the band number.


=======================  ===============================================
Object Name              Description
=======================  ===============================================
f_wr10                   WR-10, 75-110 GHz
f_wr3                    WR-3, 220-325 GHz
f_wr2p2                  WR-2.2, 330-500 GHz
f_wr1p5                  WR-1.5, 500-750 GHz
f_wr1                    WR-1, 750-1100 GHz
=======================  ===============================================


:class:`~skrf.media.media.Media` Objects
==============================================
These are predefined :class:`~skrf.media.media.Media` objects
that represent Standardized transmission line media's. This information

Rectangular Waveguide Media's
++++++++++++++++++++++++++++++++++

:class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`
Objects for standard bands.

=======================  ===============================================
Object Name              Description
=======================  ===============================================
wr10                     WR-10, 75-110 GHz
wr3                      WR-3, 220-325 GHz
wr2p2                    WR-2.2, 330-500 GHz
wr1p5                    WR-1.5, 500-750 GHz
wr1                      WR-1, 750-1100 GHz
=======================  ===============================================



References
-------------
.. [#] VDI Application Note:  VDI Waveguide Band Designations (VDI-1002) http://vadiodes.com/VDI/pdf/waveguidechart200908.pdf
'''


from network import *
from frequency import Frequency
from media import RectangularWaveguide

import warnings
import os
import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime


# pre-initialized classes

#frequency bands
f_wr10  = Frequency(75,110,201, 'ghz')
f_wr3   = Frequency(220,325,201, 'ghz')
f_wr2p2 = Frequency(330,500,201, 'ghz')
f_wr1p5 = Frequency(500,750,201, 'ghz')
f_wr1   = Frequency(750,1100,201, 'ghz')

# rectangular waveguides
wr10    = RectangularWaveguide(Frequency(75,110,201, 'ghz'), 100*mil,z0=50)
wr3     = RectangularWaveguide(Frequency(220,325,201, 'ghz'), 30*mil,z0=50)
wr2p2   = RectangularWaveguide(Frequency(330,500,201, 'ghz'), 22*mil,z0=50)
wr1p5   = RectangularWaveguide(Frequency(500,750,201, 'ghz'), 15*mil,z0=50)
wr1     = RectangularWaveguide(Frequency(750,1100,201, 'ghz'), 10*mil,z0=50)



## Functions
# Ploting
def save_all_figs(dir = './', format=['eps','pdf','png']):
    '''
    Save all open Figures to disk.

    Parameters
    ------------
    dir : string
            path to save figures into
    format : list of strings
            the types of formats to save figures as. The elements of this
            list are passed to :matplotlib:`savefig`. This is a list so that
            you can save each figure in multiple formats.
    '''
    if dir[-1] != '/':
        dir = dir + '/'
    for fignum in plb.get_fignums():
        fileName = plb.figure(fignum).get_axes()[0].get_title()
        if fileName == '':
            fileName = 'unamedPlot'
        for fmt in format:
            plb.savefig(dir+fileName+'.'+fmt, format=fmt)
            print (dir+fileName+'.'+fmt)

def add_markers_to_lines(ax=None,marker_list=['o','D','s','+','x'], markevery=10):
    if ax is None:
        ax=plb.gca()
    lines = ax.get_lines()
    if len(lines) > len (marker_list ):
        marker_list *= 3
    [k[0].set_marker(k[1]) for k in zip(lines, marker_list)]
    [line.set_markevery(markevery) for line in lines]

def legend_off(ax=None):
    '''
    turn off the legend for a given axes. if no axes is given then
    it will use current axes.
    '''
    if ax is None:
        plb.gca().legend_.set_visible(0)
    else:
        ax.legend_.set_visible(0)

def plot_complex(z,*args, **kwargs):
    '''
    plots a complex array or list in real vs imaginary.
    '''
    plb.plot(npy.array(z).real,npy.array(z).imag, *args, **kwargs)


# other
def now_string():
    return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')

def find_nearest(array,value):
    '''
    find nearest value in array.
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    '''
    idx=(npy.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    '''
    find nearest value in array.
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    '''
    return (npy.abs(array-value)).argmin()

## file conversion
def statistical_2_touchstone(file_name, new_file_name=None,\
        header_string='# GHz S RI R 50.0'):
    '''
    converts the file format used by Statistical and other Dylan Williams
    software to standard touchstone format.

    Parameters
    ------------
    file_name : string
            name of file to convert
    new_file_name : string
            name of new file to write out (including extension)
    header_string : string
            touchstone header written to first beginning of file

    '''
    old_file = file(file_name,'r')
    new_file = open(new_file_name,'w')
    new_file.write('%s\n'%header_string)
    for line in old_file:
        new_file.write(line)
    new_file.close()
    old_file.close()



## script templates
script_templates = {}
script_templates['cal_gen_ideals'] = \
'''
from pylab import *
from scipy.constants import *
import skrf as rf

################################# INPUT ################################
measured_dir = ''
media_type = ''
media_kwargs ={}
f_unit = 'ghz'
########################################################################


measured_dict = rf.load_all_touchstones(measured_dir,f_unit=f_unit)
frequency = measured_dict.values()[0].frequency
media = rf.__getattribute__(media_type)(frequency, **media_kwargs)

cal = rf.Calibration(
        ideals = [
                media.(),
                media.(),
                media.(),
                ],
        measured = [
                measured_dict[''],
                measured_dict[''],
                measured_dict[''],
                ]
        )
'''
script_templates['cal'] = \
'''
from pylab import *
from scipy.constants import *
import skrf as rf

################################# INPUT ################################
measured_dir = ''
ideals_dir = ''
ideals_names = ['','','']
measured_names = ['','','']
f_unit = 'ghz'
########################################################################


measured_dict = rf.load_all_touchstones(measured_dir,f_unit=f_unit)
ideals_dict = rf.load_all_touchstones(measured_dir,f_unit=f_unit)
frequency = measured_dict.values()[0].frequency
[ideals_dict[k].resample(frequency.npoints) for k in ideals_dict]

cal = rf.Calibration(
        ideals = [ideals_dict[k] for k in ideals_names],
        measured = [measured_dict[k] for k in measured_names],
        )
'''

def script_template(template_name, file_name='skrf_script.py', \
        overwrite=False, *args, **kwargs):
    '''
    creates skrf scripts based on templates

    Parameters
    -----------
    template_name : string ['cal', 'cal_gen_ideals']
            name of template to use
    file_name : string
            name of script file to write
    overwrite : Boolean
            if file_name exists should it be overwritten
    \*args, \*\*kwargs : arguments and keyword arguments
            passed to open()
    '''
    if template_name not in script_templates.keys():
        raise(ValueError('\'%s\' not valid template_name'%template_name))

    if os.path.isfile(file_name) and overwrite is False:
        warnings.warn('%s exists, and `overwrite` is False, abort. '\
                %file_name)
    else:
        script_file = open(file_name, 'w')
        script_file.write(script_templates[template_name])
        script_file.close()
