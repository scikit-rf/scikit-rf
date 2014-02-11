
from traits.api import  *

from .constants import ALMOST_ZERO, ALMOST_INF, ALMOST_ONE
from .frequency import Frequency
from .mathFunctions import complex_2_db, complex_2_db10, complex_2_degree
from .plotting import plot_complex_rectangular,plot_rectangular

import numpy as npy
import pylab as plb

class Projection(object):
    '''
    a scalar projection of a parameter
    '''
    def __init__(self,network, param, func):
        self.network = network
        self.param = param
        self.func = func
    
    @property
    def val(self):
        return self.func(self.param.val)
    
    def plot(self):
        
        return plot_rectangular(self.network.frequency.f_scaled, 
                                self.val.flatten())
    

class Param(object):
    '''
    a complex network parameter
    '''
    def __init__(self,  network, from_s, **kwargs):
        self.from_s = from_s
        self.network = network
    
    
    @property
    def val(self):
        return self.from_s(self.network.s.val)
    
        
    @property
    def db10(self):
        return Projection(self.network, self,complex_2_db10)
    
    def plot(self):
        return plot_complex_rectangular(self.val.flatten())
    

class S(Param):
    '''
    s parameters 
    
    these are special, because they are the internal storage format 
    
    '''
    def __init__(self,  network, s):
        
        s_shape= npy.shape(s)
        if len(s_shape) <3:
            if len(s_shape) == 2:
                # reshape to kx1x1, this simplifies indexing in function
                s = npy.reshape(s,(-1,s_shape[0],s_shape[0]))
            else:
                s = npy.reshape(s,(-1,1,1))

        self._val= npy.array(s,dtype=complex)
        self.network = network
        
        
    
    @property
    def val(self):
        return self._val
    
    
    
    
    
class Network(object):
    def __init__(self, s, frequency):
        
        self.frequency = frequency
        self.s = S(self, s)
        self.z = Param(self, s2z)

    


       
def fix_z0_shape( z0, nfreqs, nports):
    '''
    Make a port impedance of correct shape for a given network's matrix 
    
    This attempts to broadcast z0 to satisy
        npy.shape(z0) == (nfreqs,nports)
    
    Parameters 
    --------------
    z0 : number, array-like
        z0 can be: 
        * a number (same at all ports and frequencies)
        * an array-like of length == number ports.
        * an array-like of length == number frequency points.
        * the correct shape ==(nfreqs,nports)
    
    nfreqs : int
        number of frequency points
    nportrs : int
        number of ports
        
    Returns
    ----------
    z0 : array of shape ==(nfreqs,nports)
        z0  with the right shape for a nport Network

    Examples
    ----------
    For a two-port network with 201 frequency points, possible uses may
    be
    
    >>> z0 = rf.fix_z0_shape(50 , 201,2)
    >>> z0 = rf.fix_z0_shape([50,25] , 201,2)
    >>> z0 = rf.fix_z0_shape(range(201) , 201,2)

        
    '''
    
    
    
    if npy.shape(z0) == (nfreqs, nports):
        # z0 is of correct shape. super duper.return it quick.
        return z0.copy() 
    
    elif npy.isscalar(z0):
        # z0 is a single number
        return npy.array(nfreqs*[nports * [z0]])
    
    elif len(z0)  == nports:
        # assume z0 is a list of impedances for each port, 
        # but constant with frequency 
        return npy.array(nfreqs*[z0])
        
    elif len(z0) == nfreqs:
        # assume z0 is a list of impedances for each frequency,
        # but constant with respect to ports
        return npy.array(nports * [z0]).T
        
    else: 
        raise IndexError('z0 is not acceptable shape')

## network parameter conversion       
def s2z(s,z0=50):
    '''
    Convert scattering parameters [1]_ to impedance parameters [2]_


    .. math::
        z = \\sqrt {z_0} \\cdot (I + s) (I - s)^{-1} \\cdot \\sqrt{z_0}

    Parameters
    ------------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number 
        port impedances.                                         

    Returns
    ---------
    z : complex array-like
        impedance parameters

    
        
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/S-parameters
    .. [2] http://en.wikipedia.org/wiki/impedance_parameters
    
    '''
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    
    z = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    s = s.copy() # to prevent the original array from being altered
    s[s==1.] = 1. + 1e-12 # solve numerical singularity
    s[s==-1.] = -1. + 1e-12 # solve numerical singularity
    for fidx in xrange(s.shape[0]):
        sqrtz0 = npy.mat(npy.sqrt(npy.diagflat(z0[fidx])))
        z[fidx] = sqrtz0 * (I-s[fidx])**-1 * (I+s[fidx]) * sqrtz0
    return z
