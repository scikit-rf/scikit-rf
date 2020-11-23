import numpy as npy
from ..noisyNetwork import NoisyNetwork, cascade_2port, parallel_parallel_2port, series_series_2port
from ..networkNoiseCov import NetworkNoiseCov
from ..util import network_array
from .. import noisyComponents
from skrf.constants import *
import math

class SmallSig_NPN_BJT_CE_2port(NoisyNetwork):
    """ Small-signal hybrid-pi Ebers-Moll BJT model

    See "RF Circuit Design, Theory and Applications" - Ludwig page 366

    This is the model that SPICE uses: https://www.andyc.diy-audio-engineering.org/spice_models_3.html

    Args:
        Network ([type]): [description]
    """

    def __init__(self, ic_ma=0, beta0=100, cpi=1e-18, cu=1e-18, rbb=0, rc=0, re=0, ru=0, r0=0, cjx=1e-20, cjs=1e-20, name=None, comments=None, f_unit=None, T0=None, s_def='power', **kwargs):
        super().__init__(name=name, comments=comments, f_unit=f_unit, s_def=s_def, **kwargs)

        nwrk_rbb = noisyComponents.RLC_Series_2port(R=rbb, frequency=self.frequency, T0 = T0)
        nwrk_rc = noisyComponents.RLC_Series_2port(R=rc, frequency=self.frequency, T0 = T0)
        nwrk_re = noisyComponents.RLC_Shunt_2port(R=re, frequency=self.frequency, T0 = T0)
        nwrk_ru = noisyComponents.RLC_Series_2port(R=ru, frequency=self.frequency, T0 = T0)
        nwrk_r0 = noisyComponents.RLC_Shunt_2port(R=r0, frequency=self.frequency, T0 = T0)

        nwrk_cjx = noisyComponents.RLC_Series_2port(C=cjx, frequency=self.frequency, T0 = T0)
        nwrk_cjs = noisyComponents.RLC_Shunt_2port(C=cjs, frequency=self.frequency, T0 = T0)

        if T0:
            self.T0 = T0
        Vt = K_BOLTZMANN*self.T0/Q_CHARGE # thermal voltage

        gm = ic_ma/Vt/1000. # transconductance
        rpi = beta0/gm # input resistance

        nwrk_rpi_cpi = noisyComponents.RLC_Shunt_2port(R=rpi, C=cpi, frequency=self.frequency, T0 = T0)
        nwrk_rpi_cpi.noise_source('none') # transconductance not responsible for thermal noise

        scu = 1j*self.frequency.w*cu
        sug_d = scu - gm
        cug_shfb_a = network_array([[scu/sug_d,     1/sug_d],
                                    [scu*gm/sug_d,   scu/sug_d]])
        nwrk_inner_bjt = NoisyNetwork.from_a(cug_shfb_a, frequency=self.frequency)
        nwrk_inner_bjt.noise_source('none')

        ntwk_bjt_shot_noise = cascade_2port(nwrk_rpi_cpi, nwrk_inner_bjt)

        # Model the shot noise using y form of the covariance matrix
        ovec = npy.ones(len(self.frequency))
        zvec = npy.zeros(len(self.frequency))
        cov_bjt_y = 2*K_BOLTZMANN*self.T0*gm*network_array([[1/beta0*ovec,  zvec],
                                                            [zvec,          ovec]])

        ntwk_bjt_shot_noise.noise_source(NetworkNoiseCov(cov_bjt_y, form='y'))

        ntwkT = parallel_parallel_2port(ntwk_bjt_shot_noise, nwrk_ru)
        ntwkT = cascade_2port(nwrk_rbb, ntwkT)
        ntwkT = cascade_2port(ntwkT, nwrk_r0)
        ntwkT = series_series_2port(ntwkT, nwrk_re)
        ntwkT = parallel_parallel_2port(ntwkT, nwrk_cjx)
        ntwkT = cascade_2port(ntwkT, nwrk_cjs)
        ntwkT = cascade_2port(ntwkT, nwrk_rc)

        self.s = ntwkT.s
        self.noise_source(ntwkT.noise_cov, T0=T0)

