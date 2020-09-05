import numpy as npy
from ..network import Network
from ..util import network_array
import math

class RLC_Shunt_2port(Network):

    def __init__(self, R=math.inf, L=math.inf, C=0, name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        w = self.frequency.w
        if L == math.inf:
            if R == math.inf:
                if C == 0:
                    Z = math.inf
                else:
                    Z = 1/(1j*w*C)
            else:
                Z = R/(1 + 1j*w*R*C)
        elif R == math.inf:
            Z = 1j*w*L/(1 - w*w*L*C)
        else:
            Z = 1j*w*L*R/(R + 1j*w*L*(1 + 1j*w*R*C))

        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))

        if L == math.inf and R == math.inf and C == 0:
            unity = network_array([[zvec,ovec],
                                   [ovec,zvec]])
            self.s = unity
            self.noise_source('passive', T0)
        else:
            shunt = network_array([[Z, Z],
                                  [Z, Z]])
            self.s = npy.zeros(shape=shunt.shape)
            self.z = shunt
            self.noise_source('passive', T0)