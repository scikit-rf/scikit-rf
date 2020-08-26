
import numpy as npy
from ..network import Network
from ..util import network_array

class Wilkinson_3port(Network):

    #Ideal wilkinson with isolated output ports 2 and 3 with port 1 as common port
    def __init__(self, div_dB_p2, phase_p2, div_dB_p3, phase_p3, name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        if div_dB_p2 < 0 or div_dB_p3 < 0:
            raise ValueError("loss_db must be a positive value")

        self._S12 = (10**(-div_dB_p2/20))*npy.exp(1j*phase_p2*(npy.pi/180))
        self._S13 = (10**(-div_dB_p3/20))*npy.exp(1j*phase_p3*(npy.pi/180))
        

        if npy.abs(self._S12)**2 + npy.abs(self._S13)**2 > 1:
            raise ValueError("Wilkinson powersplit is non-physical")

        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))

        dev_vec  = network_array([[zvec,                             self._S12*ovec,                   self._S13*ovec], \
                                  [self._S12*ovec,                             zvec,                             zvec], \
                                  [self._S13*ovec,                             zvec,                             zvec]])

        self.s = div_vec
        self.noise_source('passive', T0)




