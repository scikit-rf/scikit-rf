

'''

.. currentmodule:: skrf.constants
========================================
constants (:mod:`skrf.constants`)
========================================

This module contains constants, numerical approximations, and unit conversions


'''

from scipy.constants import c, micron, mil, inch, centi, milli, nano, micro,pi



# used as substitutes to handle mathematical singularities.
INF = 1e99
ONE = 1.0 + 1/1e14
ZERO = 1e-6

def to_meters( d, unit='m'):
    '''
    Translate various  units of distance into meters

    

    Parameters
    ------------
    d : number or array-like
        the value
    unit : str
        the unit to that x is in:
        ['m','cm','um','in','mil','s','us','ns','ps']

    '''
    unit = unit.lower()
    d_dict ={'m':d,
             'cm':1e-2*d,
             'mm':1e-3*d,
             'um':1e-6*d,
             'in':d*inch,
             'mil': d*mil,
             's':d*c,
             'us':d*1e-6*c,
             'ns':d*1e-9*c,
             'ps':d*1e-12*c,
             }
    try:
        return d_dict[unit]
    except(KeyError):
        raise(ValueError('Incorrect unit'))

