
from traits.api import  *

from .constants import ALMOST_ZERO, ALMOST_INF, ALMOST_ONE
from .frequency import Frequency
from .mathFunctions import complex_2_db, complex_2_db10, complex_2_degree
from .plotting import plot_complex_rectangular,plot_rectangular

import numpy as npy
import pylab as plb

from IPython.display import Image, SVG, Math
from IPython.core.pylabtools import print_figure




class Projection(object):
    '''
    a scalar projection of a parameter
    '''
    def __init__(self,network, param, func, name, y_label):
        self.network = network
        self.param = param
        self.func = func
        self.name = name
        self.y_label = y_label
        self._png_data = None
        self._svg_data = None
        
        
    @property
    def val(self):
        return self.func(self.param.val)
    
    def plot(self,  m=None, n=None, ax=None, show_legend=True,*args, 
             **kwargs):

        # create index lists, if not provided by user
        if m is None:
            M = range(self.network.nports)
        else:
            M = [m]
        if n is None:
            N = range(self.network.nports)
        else:
            N = [n]

        if 'label'  not in kwargs.keys():
            gen_label = True
        else:
            gen_label = False

        lines = [] #list of mpl lines
        for m in M:
            for n in N:
                # set the legend label for this trace to the networks
                # name if it exists, and they didnt pass a name key in
                # the kwargs
                if gen_label:
                    if self.name is None:
                        if plb.rcParams['text.usetex']:
                            label_string = '$%s_{%i%i}$'%\
                            (self.param.name.upper(),m+1,n+1)
                        else:
                            label_string = '%s%i%i'%\
                            (self.param.name.upper(),m+1,n+1)
                    else:
                        if plb.rcParams['text.usetex']:
                            label_string = self.network.name+', $%s_{%i%i}$'%\
                            (self.param.name.upper(),m+1,n+1)
                        else:
                            label_string = self.network.name+', %s%i%i'%\
                            (self.param.name.upper(),m+1,n+1)
                    kwargs['label'] = label_string

                # plot the desired attribute vs frequency
                if 'time' in self.param.name: 
                    x_label = 'Time (ns)'
                    x = self.network.frequency.t_ns
                    
                else:
                    x_label = 'Frequency (%s)'%self.network.frequency.unit
                    x = self.network.frequency.f_scaled
                
                
                lines.append(plot_rectangular(
                        x = x,
                        y = self.val[:,m,n],
                        x_label = x_label,
                        y_label = self.y_label,
                        show_legend = show_legend, ax = ax,
                        *args, **kwargs)[0])
        return lines
    
    def _figure_data(self, format):
        fig, ax = plb.subplots()
        self.plot(ax=ax)
        data = print_figure(fig, format)
        plb.close(fig)
        return data
    
    def _repr_png_(self):
        if self._png_data is None:
            self._png_data = self._figure_data('png')
        return self._png_data
    
    @property
    def png(self):
        return Image(self._repr_png_(), embed=True)
        
class Parameter(object):
    '''
    a complex network parameter
    '''
    def __init__(self,  network,  name, **kwargs):
        self.name = name
        self.network = network
        self._png_data = None
        self._svg_data = None
    
    @property
    def val(self):
        raise NotImplementedError('Subclass Must implement me')
    
    @property
    def db10(self):
        return Projection(self.network, self,complex_2_db10, 
                          name= 'dB10',y_label='Magnitude (dB)')
    
    @property
    def deg(self):
        return Projection(self.network, self,complex_2_degree, 
                          name= 'deg',y_label='Phase (deg)')
    
    def plot(self, m=None, n=None, ax=None, show_legend=True,*args, 
             **kwargs):

        # create index lists, if not provided by user
        if m is None:
            M = range(self.network.nports)
        else:
            M = [m]
        if n is None:
            N = range(self.network.nports)
        else:
            N = [n]

        if 'label'  not in kwargs.keys():
            gen_label = True
        else:
            gen_label = False

        
        #was_interactive = plb.isinteractive
        #if was_interactive:
        #    plb.interactive(False)
        lines = []
        for m in M:
            for n in N:
                # set the legend label for this trace to the networks
                # name if it exists, and they didnt pass a name key in
                # the kwargs
                if gen_label:
                    if self.name is None:
                        if plb.rcParams['text.usetex']:
                            label_string = '$%s_{%i%i}$'%\
                            (self.name.upper(),m+1,n+1)
                        else:
                            label_string = '%s%i%i'%\
                            (self.name.upper(),m+1,n+1)
                    else:
                        if plb.rcParams['text.usetex']:
                            label_string = self.network.name+', $%s_{%i%i}$'%\
                            (self.name.upper(),m+1,n+1)
                        else:
                            label_string = self.network.name+', %s%i%i'%\
                            (self.name.upper(),m+1,n+1)
                    kwargs['label'] = label_string

                # plot the desired attribute vs frequency
                lines.append(plot_complex_rectangular(
                    z = self.val[:,m,n],
                    show_legend = show_legend, ax = ax,
                    *args, **kwargs)[0])
        return lines
    
    def _figure_data(self, format):
        fig, ax = plb.subplots()
        self.plot(ax=ax)
        data = print_figure(fig, format)
        plb.close(fig)
        return data
    
    def _repr_png_(self):
        if self._png_data is None:
            self._png_data = self._figure_data('png')
        return self._png_data
    
    @property
    def png(self):
        return Image(self._repr_png_(), embed=True)
    
    
class S(Parameter):
    '''
    s parameters 
    
    these are special, because they are the internal storage format 
    
    '''
    def __init__(self,  network, s):
        
        s_shape= npy.shape(s)
        # shaping array 
        if len(s_shape) <3:
            if len(s_shape) == 2:
                # reshape to kx1x1, this simplifies indexing in function
                s = npy.reshape(s,(-1,s_shape[0],s_shape[0]))
            else:
                s = npy.reshape(s,(-1,1,1))

        self._val= npy.array(s,dtype=complex)
        self.network = network
        self.name = 's'
        self._png_data = None
        self._svg_data = None
    
    @property
    def val(self):
        return self._val


class Z(Parameter):
    @property
    def val(self):
        return s2z(self.network.s.val)

class Network(object):
    def __init__(self, s, frequency, name = ''):
        self.frequency = frequency
        self.s = S(self, s,name = 's')
        self.z = Z(self, name='z')
        self.name = name
    
    @classmethod
    def from_z(cls, z, z0=50, **kwargs):
        return cls(s = z2s(z,z0), **kwargs)
        
    @classmethod
    def from_old_Network( cls, network):
        return cls(frequency = network.frequency,
                   s = network.s,
                   name = network.name,
                   )
        
        
    @property
    def nports(self):
        '''
        the number of ports the network has.

        Returns
        --------
        nports : number
                the number of ports the network has.

        '''
        try:
            return self.s.val.shape[1]
        except (AttributeError):
            return 0


       
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
