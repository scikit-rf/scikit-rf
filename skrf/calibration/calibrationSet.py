
#       calibrationSet.py
#
#
#       Copyright 2012 alex arsenovic <arsenovic@virginia.edu>
#
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
.. module:: skrf.calibration.calibrationSet
================================================================
calibrationSet (:mod:`skrf.calibration.calibrationSet`)
================================================================


Contains the CalibrationSet class, and supporting functions

CalibrationSet Class
===============

.. autosummary::
   :toctree: generated/

   CalibrationSet

'''
from itertools import product, combinations, permutations
from calibration import Calibration
from ..networkSet import NetworkSet

def cartesian_product(ideals, measured_sets, *args, **kwargs):
    '''
    '''
    measured_lists = product(*[k[:] for k in measured_sets])
    return [Calibration(ideals = ideals, measured = measured, *args, **kwargs) \
        for measured in measured_lists ]


class CalibrationSet(object):
    '''
    '''
    combinitoric_func_dict = {
        'cartesian': cartesian_product,
        }

    def __init__(self, ideals, measured_sets, type='cartesian',
        *args, **kwargs):
        '''
        
        '''
        self.ideals = ideals
        self.measured_sets = measured_sets
        self.type = type
        self.args = args
        self.kwargs = kwargs
        self.cal_list = None
        self.run()

    def apply_cal(self, raw_ntwk, *args, **kwargs):
        '''
        '''
        return NetworkSet([k.apply_cal(raw_ntwk) for k in self.cal_list],
            *args, **kwargs)

    def plot_uncertainty_per_standard(self):
        '''
        '''
        self.dankness('std_s','plot_s_mag')

    def dankness(self, prop, func, *args, **kwargs):
        '''
        '''
        try:
            [k.__getattribute__(prop).__getattribute__(func)\
                (*args, **kwargs) for k in self.measured_sets]
        except (TypeError):
            return [k.__getattribute__(prop).__getattribute__(func) \
                for k in self.measured_sets]
                
    def run(self):
        self.cal_list = self.combinitoric_func_dict[self.type](
            ideals = self.ideals,
            measured_sets = self.measured_sets,
            *self.args, **self.kwargs)
    
