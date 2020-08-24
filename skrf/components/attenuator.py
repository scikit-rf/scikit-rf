
import numpy as npy
from ..network import Network
from ..util import network_array

class Attenuator(Network):

    def __init__(self, attenuation_db, name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        if attenuation_db < 0:
            raise ValueError("attenuation_db must be a positive value")
        
        self._attenuation = 10**(-attenuation_db/20)

        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))

        attn_vec = network_array([[zvec,                   self._attenuation*ovec],
                                     [self._attenuation*ovec,                    zvec]])

        self.s = attn_vec
        self.noise_source('passive', T0)




