from ..frequency import *
from ..network import *

class Deembedding(object):
    '''
    Base Class for all de-embedding objects
    '''
    def __init__(self, dummy, name=None, *args, **kwargs):
        '''
        De-embedding Initializer

        Notes
        -----
        Parameters
        ----------
        dummies: network objects of dummy structures used for de-embedding 
        '''

        # fill dummy with copied lists of input
        self.dummy = [ntwk.copy() for ntwk in dummy]

       # ensure all the dummy Networks' frequency's are the same
        for dmyntwk in self.dummy:
            if self.dummy[0].frequency != dmyntwk.frequency:
                raise(ValueError('Dummy Networks dont have matching frequencies.')) 

        # may attempt to interpolate if frequencies do not match

        self.kwargs = kwargs
        self.name = name

    def __str__():
        pass

    def __repr_():
        pass

    def apply_cal(self,ntwk):
        '''
        Apply correction to a Network
        '''
        raise NotImplementedError('The Subclass must implement this')

class OpenShort(Deembedding):
    '''
    2-step open-short de-embedding [1]_

    [1] M. C. A. M. Koolen, J. A. M. Geelen and M. P. J. G. Versleijen, "An improved 
    de-embedding technique for on-wafer high frequency characterization", 
    IEEE 1991 Bipolar Circuits and Technology Meeting, pp. 188-191, Sep. 1991.
    '''
    def __init__(self, measured, *args, **kwargs):
        '''
        Docstring
        '''
        Deembedding.__init__(self, measured, *args, *kwargs)

    def apply_cal(self, ntwk):
        '''
        Docstring
        '''
       
        # first measured ntwk is open dummy
        open = self.dummy[0].copy()

        # second measured ntwk is short dummy
        short = self.dummy[1].copy()
        
        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != open.frequency:
            raise(ValueError('Network frequencies dont match dummy frequencies.')) 

        caled = ntwk.copy()

        # remove open parasitics
        caled.y = ntwk.y - open.y
        # remove short parasitics
        caled.z = caled.z - short.z

        return caled
