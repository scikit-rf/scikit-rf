
import numpy as npy
from ..noisyNetwork import NoisyNetwork
from ..util import network_array

class Wilkinson_Nport(NoisyNetwork):

    #Ideal wilkinson with isolated output ports 2 and 3 with port 1 as common port
    def __init__(self, N_outs, loss_db = 0, phases = [], name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        if loss_db < 0:
            raise ValueError("loss_db must be a positive value")

        if not phases:
            self._loss = (10**(-loss_db/20))
        else:
            if len(phases) <  N_outs:
                raise ValueError("phases must be a 1xN_outs array of phases")

            self._loss = self._loss * npy.exp(1j*phases*(npy.pi/180))
        
        alpha = (1/npy.sqrt(N_outs))

        nway_mat = npy.zeros((len(self.frequency),N_outs+1, N_outs+1)) 

        nway_mat[:,0,1:] = self._loss * alpha * npy.ones((len(self.frequency),N_outs))
        nway_mat[:,1:,0] = self._loss * alpha * npy.ones((len(self.frequency),N_outs))

        self.s = nway_mat
        self.noise_source('passive', T0)




