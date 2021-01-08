import numpy as npy
from ..noisyNetwork import NoisyNetwork
from ..util import network_array
import math

class RLC_Series_2port(NoisyNetwork):

    def __init__(self, R=0, L=0, C=math.inf, name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        w = self.frequency.w
        if C == math.inf:
            Z = R + 1j*w*L
        else:
            Z = R + 1j*w*L + 1/(1j*w*C)

        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))

        if R == 0 and L == 0 and C==math.inf:
            unity = network_array([[zvec,ovec],
                                   [ovec,zvec]])
            self.s = unity
            self.noise_source('passive', T0)
        else:
            series = network_array([[1/Z, -1/Z],
                                    [-1/Z, 1/Z]])
            
            self.s = npy.zeros(shape=series.shape)
            self.y = series
            self.noise_source('passive', T0)