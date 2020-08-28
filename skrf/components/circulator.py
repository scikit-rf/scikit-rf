
import numpy as npy
from ..network import Network
from ..util import network_array

class Circulator(Network):

    def __init__(self, loss_db, iso_db = 10000,  name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        if loss_db < 0:
            raise ValueError("loss_db must be a positive value")
        if iso_db < 0:
            raise ValueError("iso_db must be a positive value")
        
        self._attenuation = 10**(-loss_db/20)
        self._isolation   = 10**(-iso_db/20)

        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))

        circ_vec = network_array([[zvec,                      self._isolation*ovec,            self._attenuation*ovec], \
                                  [self._attenuation*ovec,                    zvec,              self._isolation*ovec], \
                                  [self._isolation*ovec,    self._attenuation*ovec,                              zvec]])

        self.s = circ_vec
        self.noise_source('passive', T0)




