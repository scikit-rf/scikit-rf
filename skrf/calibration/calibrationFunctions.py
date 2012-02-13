
#       calibrationFunctions.py
#
#
#       Copyright 2011 alex arsenovic <arsenovic@virginia.edu>
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
.. module:: skrf.calibration.calibrationFunctions
===================================================================================
calibrationFunctions (:mod:`skrf.calibration.calibrationFunctions`)
===================================================================================



Functions which operate on or pertain to :class:`~skrf.calibration.calibration.Calibration` Objects

.. autosummary::
   :toctree: generated/

        cartesian_product_calibration_set

'''
from itertools import product, combinations, permutations
from calibration import Calibration
from numpy import array

def cartesian_product_calibration_set( ideals, measured, *args, **kwargs):
    '''
    This function is used for calculating calibration uncertainty due
    to un-biased, non-systematic errors.

    It creates an ensemble of calibration instances. the set  of
    measurement lists used in the ensemble is the Cartesian Product
    of all instances of each measured standard.

    The idea is that if you have multiple measurements of each standard,
    then the multiple calibrations can be made by generating all possible
    combinations of measurements.  This produces a conceptually simple,
    but computationally expensive way to estimate calibration uncertainty.



    takes:
            ideals: list of ideal Networks
            measured: list of measured Networks
            *args,**kwargs: passed to Calibration initializer

    returns:
            cal_ensemble: a list of Calibration instances.


    you can use the output to estimate uncertainty by calibrating a DUT
    with all calibrations, and then running statistics on the resultant
    set of Networks. for example

    import skrf as rf
    # define you lists of ideals and measured networks
    cal_ensemble = \
            rf.cartesian_product_calibration_ensemble( ideals, measured)
    dut = rf.Network('dut.s1p')
    network_ensemble = [cal.apply_cal(dut) for cal in cal_ensemble]
    rf.plot_uncertainty_mag(network_ensemble)
    [network.plot_s_smith() for network in network_ensemble]
    '''
    measured_iterable = \
            [[ measure for measure in measured \
                    if ideal.name in measure.name] for ideal in ideals]
    measured_product = product(*measured_iterable)

    return [Calibration(ideals =ideals, measured = list(product_element),\
            *args, **kwargs)\
            for product_element in measured_product]

def dot_product_calibration_set( ideals, measured, *args, **kwargs):
    '''
    This function is used for calculating calibration uncertainty due
    to un-biased, non-systematic errors.

    It creates an ensemble of calibration instances. the set  of
    measurement lists used in the ensemble is the python 'zip' of measured
    and ideal lists. it is equivalent to making a calibration for each
    connection instance of a measurement.

    This requires the same number of repeat elements for

    takes:
            ideals: list of ideal Networks
            measured: list of measured Networks
            *args,**kwargs: passed to Calibration initializer

    returns:
            cal_ensemble: a list of Calibration instances.
    '''
    # this is a way to sort the measured list by alphabetically ordering
    # of the Network element names
    measured_range = range(len(measured))
    name_list = [k.name for k in measured]
    sorted_index = sorted(measured_range, key = lambda k:name_list[k])
    measured = [measured[k] for k in sorted_index]

    measured_iterable = \
            [[ measure for measure in measured \
                    if ideal.name in measure.name] for ideal in ideals]
    m_array= array( measured_iterable)
    return [Calibration(ideals = ideals, measured = list(m_array[:,k]),\
            *args, **kwargs)\
            for k in range(m_array.shape[1])]

def binomial_coefficient_calibration_set( ideals, measured, n,  *args, **kwargs):
    '''
    Produces a ensemble of calibration instances based on choosing
    sub-sets of the ideal/measurement lists from an overdetermined
    calibration. This concept is described in 'De-embeding and
    Un-terminating' by Penfield and Baurer.

    so, if the calibration ideals and measured lists have length 'm'
    then the resultant ensemble of calibrations is 'm choose n' long.


    takes:
            ideals: list of ideal Networks
            measured: list of measured Networks
            n: length of ideal/measured lists to pass to calibrations
                    (must be < len(ideals) )
            *args,**kwargs: passed to Calibration initializer

    returns:
            cal_ensemble: a list of Calibration instances.



    '''
    if n >= len(ideals):
        raise ValueError('n must be larger than # of standards')

    ideal_subsets = \
            [ ideal_subset for ideal_subset in combinations(ideals,n)]
    measured_subsets = \
            [ measured_subset for measured_subset in combinations(measured,n)]

    return  [Calibration(ideals = list(k[0]), measured=list(k[1]),\
            *args, **kwargs) for k in zip(ideal_subsets, measured_subsets)]


# for backward compatability
zip_calibration_ensemble = dot_product_calibration_set
subset_calibration_ensemble = binomial_coefficient_calibration_set
