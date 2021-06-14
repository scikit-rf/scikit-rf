from abc import ABC, abstractmethod
from ..frequency import *
from ..network import *

class Deembedding(ABC):
    '''
    Abstract Base Class for all de-embedding objects
    '''
    def __init__(self, dummies, name=None, *args, **kwargs):
        '''
        De-embedding Initializer

        Notes
        -----
        Parameters
        ----------
        '''

       # ensure all the dummy Networks' frequency's are the same
        for dmyntwk in dummies:
            if dummies[0].frequency != dmyntwk.frequency:
                raise(ValueError('Dummy Networks dont have matching frequencies.')) 

        # TODO: attempt to interpolate if frequencies do not match

        self.kwargs = kwargs
        self.name = name

    def __str__():
        pass

    def __repr_():
        pass
    
    @abstractmethod
    def apply_cal(self, ntwk):
        '''
        Apply correction to a Network
        '''
        pass

class OpenShort(Deembedding):
    '''
    2-step open-short de-embedding [1]_

    [1] M. C. A. M. Koolen, J. A. M. Geelen and M. P. J. G. Versleijen, "An improved 
    de-embedding technique for on-wafer high frequency characterization", 
    IEEE 1991 Bipolar Circuits and Technology Meeting, pp. 188-191, Sep. 1991.
    '''
    def __init__(self, dummy_open, dummy_short, name=None, *args, **kwargs):
        '''
        Docstring
        '''
        self.open = dummy_open.copy()
        self.short = dummy_short.copy()
        dummies = [self.open, self.short]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def apply_cal(self, ntwk):
        '''
        Docstring
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
