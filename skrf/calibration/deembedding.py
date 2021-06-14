'''
.. module:: skrf:calibration:deembedding

====================================================
deembedding (:mod:`skrf.calibration.deembedding`)
====================================================

This module provides objects to implement de-embedding methods
for on-wafer applications. Each de-embedding method inherits
from the common abstract base class :class:`Deembedding`.

Base Class
---------------

.. autosummary::
    :toctree: generated/

    Deembedding

De-embedding Methods
-----------------------

.. autosummary::
    :toctree: generated/

    OpenShort
'''

from abc import ABC, abstractmethod
from ..frequency import *
from ..network import *

class Deembedding(ABC):
    '''
    Abstract Base Class for all de-embedding objects.

    This class implements the common mechanisms for all de-embedding
    algorithms. Specific calibration algorithms should inherit this
    class and over-ride the methods:
    * :func:`Deembedding.deembed`

    '''
    def __init__(self, dummies, name=None, *args, **kwargs):
        '''
        De-embedding Initializer

        Notes
        -----
        Each de-embedding algorithm may use a different number of
        dummy networks. We check that each of these dummy networks
        have matching frequecies to perform de-embedding.

        It should be known a-priori what the equivalent circuit 
        of the parasitic network looks like. The proper de-embedding
        method should then be chosen accordingly.

        Parameters
        ----------
        dummies : list of :class:`~skrf.network.Network` objects
            Network info of all the dummy structures used in a 
            given de-embedding algorithm.

        name : string
            Name of this de-embedding instance, like 'open-short-set1'
            This is for convenience of identification.

        \*args, \*\*kwargs : keyword arguments
            stored in self.args and self.kwargs, which may be used
            by sub-classes if needed.
        '''

       # ensure all the dummy Networks' frequency's are the same
        for dmyntwk in dummies:
            if dummies[0].frequency != dmyntwk.frequency:
                raise(ValueError('Dummy Networks dont have matching frequencies.')) 

        # TODO: attempt to interpolate if frequencies do not match

        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __str__():
        pass

    def __repr_():
        pass
    
    @abstractmethod
    def deembed(self, ntwk):
        '''
        Apply de-embedding orrection to a Network
        '''
        pass

class OpenShort(Deembedding):
    '''
    A widely used de-embedding method for on-wafer applications.

    Two dummy measurements `dummy_open` and `dummy_short` are required.
    When :func:`Deembedding.deembed` is applied, then Y-parameters
    of the dummy_open are subtracted from the DUT measurement, followed
    by subtraction of Z-parameters of dummy-short.

    This method of de-embedding assumes the following parasitic network

                              _______
           ------------------|_______|--------------------
          |                                               |
          |    _____     __________________      _____    |
    o--------|_____|----|Device Under Test|----|_____|---------o
         _|_            -------------------              _|_
        |   |                  __|__                    |   |
        |___|                 |_____|                   |___|
          |                      |                        |
         GND                    GND                      GND

    For more information, see [1]_

    References
    ------------

    .. [1] M. C. A. M. Koolen, J. A. M. Geelen and M. P. J. G. Versleijen, "An improved 
        de-embedding technique for on-wafer high frequency characterization", 
        IEEE 1991 Bipolar Circuits and Technology Meeting, pp. 188-191, Sep. 1991.
    '''
    def __init__(self, dummy_open, dummy_short, name=None, *args, **kwargs):
        '''
        Open-Short De-embedding Initializer

        Parameters
        -----------

        dummy_open : :class:`~skrf.network.Network` object
            Measurement of the dummy open structure

        dummy_short : :class:`~skrf.network.Network` object
            Measurement of the dummy short structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        '''
        self.open = dummy_open.copy()
        self.short = dummy_short.copy()
        dummies = [self.open, self.short]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        '''
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding
        '''
        
        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.open.frequency:
            raise(ValueError('Network frequencies dont match dummy frequencies.')) 
        
        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()

        # remove open parasitics
        caled.y = ntwk.y - self.open.y
        # remove short parasitics
        caled.z = caled.z - self.short.z

        return caled
