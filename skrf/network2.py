
from six.moves import xrange # for Python3 compatibility

from .frequency import Frequency
from .mathFunctions import *
from .plotting import plot_complex_rectangular,plot_rectangular, smith
from .util import get_fid, get_extn, find_nearest_index,slice_domain

from scipy import  signal
import numpy as npy
from numpy import fft
import matplotlib.pyplot as plb

from IPython.display import Image, SVG, Math
from IPython.core.pylabtools import print_figure

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import re



##

class Parameter(object):
    '''
    a complex network parameter
    '''

    def __init__(self,  network):
        self._network = network

    def __len__(self):
        '''
        length of frequency axis
        '''
        return len(self.val)

    def __getattr__(self,name):
        return getattr(self.val,name)

    def __getitem__(self,key):
        return self.val[key]

    @property
    def val(self):
        raise NotImplementedError()

    @property
    def _xaxis(self):return 'frequency'

    ## projections
    @property
    def re(self): return Re(self)
    @property
    def im(self): return Im(self)
    @property
    def mag(self): return Mag(self)
    @property
    def deg(self): return Deg(self)
    @property
    def rad(self): return Rad(self)
    @property
    def db10(self): return Db10(self)
    @property
    def db20(self): return Db20(self)
    @property
    def db(self): return Db20(self)

    def plot(self, m=None, n=None, ax=None, show_legend=True,*args,
             **kwargs):

        # create index lists, if not provided by user
        if m is None:
            M = range(self._network.nports)
        else:
            M = [m]
        if n is None:
            N = range(self._network.nports)
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
                # name if it exists, and they didn't pass a name key in
                # the kwargs
                if gen_label:
                    if self._network.name is None:
                        if plb.rcParams['text.usetex']:
                            label_string = '$%s_{%i%i}$'%\
                            (str(self).upper(),m+1,n+1)
                        else:
                            label_string = '%s%i%i'%\
                            (str(self).upper(),m+1,n+1)
                    else:
                        if plb.rcParams['text.usetex']:
                            label_string = str(self._network)+', $%s_{%i%i}$'%\
                            (str(self).upper(),m+1,n+1)
                        else:
                            label_string = self._network.name+', %s%i%i'%\
                            (str(self).upper(),m+1,n+1)
                    kwargs['label'] = label_string

                # plot the desired attribute vs frequency
                lines.append(plot_complex_rectangular(
                    z = self.val[:,m,n],
                    show_legend = show_legend, ax = ax,
                    *args, **kwargs))#[0]) ## fix
        #return lines ## fix
    def plot_smith(self, **kwargs):
        self.plot(**kwargs)
        smith()


    ## notebook display
    def _figure_data(self, format):
        fig, ax = plb.subplots()
        self.plot(ax=ax)
        data = print_figure(fig, format)
        plb.close(fig)
        return data

    def _repr_png_(self):
        return self._figure_data('png')

    @property
    def png(self):
        return Image(self._repr_png_(), embed=True)

class S(Parameter):
    '''
    S parameters

    This Parameter is special, because they are the internal storage format

    '''
    def __init__(self,  network, s):
        Parameter.__init__(self, network)
        s = fix_parameter_shape(s)
        self._val= npy.array(s,dtype=complex)

    def __getattr__(self,name):
        return getattr(self.val,name)

    def __str__(self): return 's'

    @property
    def val(self):
        return self._val


    def plot(self, *args, **kwargs):
        out = Parameter.plot(self,*args, **kwargs)
        smith()
        return out

    def plot_complex(self, *args, **kwargs):
        return Parameter.plot(self,*args, **kwargs)

class Z(Parameter):
    '''
    Impedance parameters
    '''
    def __str__(self): return 'z'
    @property
    def val(self):
        return s2z(self._network.s.val)

class Y(Parameter):
    '''
    Admittance Parameters
    '''
    def __str__(self): return 'y'
    @property
    def val(self):
        return s2y(self._network.s.val)

class T(Parameter):
    '''
    Wave Cascading Parameters

    Only exists for 2-ports
    '''
    def __str__(self): return 't'
    @property
    def val(self):
        return s2t(self._network.s.val)

class STime(Parameter):
    '''
    Scattering Parameters in Time Domain
    '''
    def __str__(self): return 's'
    @property
    def _xaxis(self):return 'time'
    @property
    def val(self):
        return s2time(self._network.s.val)
##

class Projection(object):
    '''
    a scalar projection of a parameter
    '''
    def __init__(self, param):
        self._param = param
        self._network = param._network

    def __getitem__(self,key):
        return self.val[key]

    def __getattr__(self,name):
        return getattr(self.val,name)

    @property
    def val(self):
        raise NotImplementedError()

    def plot(self,  m=None, n=None, ax=None, show_legend=True,*args,
             **kwargs):

        # create index lists, if not provided by user
        if m is None:
            M = range(self._network.nports)
        else:
            M = [m]
        if n is None:
            N = range(self._network.nports)
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
                    if self._network.name is None:
                        if plb.rcParams['text.usetex']:
                            label_string = '$%s_{%i%i}$'%\
                            (str(self._param).upper(),m+1,n+1)
                        else:
                            label_string = '%s%i%i'%\
                            (str(self._param).upper(),m+1,n+1)
                    else:
                        if plb.rcParams['text.usetex']:
                            label_string = self._network.name+', $%s_{%i%i}$'%\
                            (str(self._param).upper(),m+1,n+1)
                        else:
                            label_string = self._network.name+', %s%i%i'%\
                            (str(self._param).upper(),m+1,n+1)
                    kwargs['label'] = label_string

                # plot the desired attribute vs frequency
                if  self._param._xaxis=='time':
                    x_label = 'Time (ns)'
                    x = self._network.frequency.t_ns

                elif self._param._xaxis=='frequency':
                    x_label = 'Frequency (%s)'%self._network.frequency.unit
                    x = self._network.frequency.f_scaled


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
        return self._figure_data('png')

    @property
    def png(self):
        return Image(self._repr_png_(), embed=True)

class Mag(Projection):
    y_label = 'Magnitude'
    unit=''
    def __str__(self):
        return ''
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)

    @property
    def val(self):
        return abs(self._param.val)

class Db10(Projection):
    y_label = 'Magnitude (dB)'
    unit='dB'
    def __str__(self):
        return 'dB'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)

    @property
    def val(self):
        return complex_2_db10(self._param.val)

class Db20(Projection):
    y_label = 'Magnitude (dB)'
    unit = 'dB'
    def __str__(self):
        return 'dB'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)
    @property
    def val(self):
        return complex_2_db(self._param.val)

class Deg(Projection):
    y_label = 'Phase (deg)'
    unit = 'deg'
    def __str__(self):
        return 'deg'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)
    @property
    def val(self):
        return complex_2_degree(self._param.val)

class Rad(Projection):
    y_label = 'Phase (rad)'
    unit = 'rad'
    def __str__(self):
        return 'rad'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)
    @property
    def val(self):
        return complex_2_radian(self._param.val)

class Re(Projection):
    y_label = 'Real Part'
    unit = ''
    def __str__(self):
        return 'real'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)
    @property
    def val(self):
        return self._param.val.real

class Im(Projection):
    y_label = 'Imaginary Part'
    unit = ''
    def __str__(self):
        return 'imag'
    def __repr__(self):
        return '{self._param}{self}'.format(self=self)
    @property
    def val(self):
        return self._param.val.imag

##

class Network(object):
    def __init__(self, frequency=None, z0=50, name='', comments='',
                 *args,  **kw):
        '''
        '''
        if 's' in kw:
            self.s = kw['s']
        elif 'z' in kw:
            self.s = z2s(kw['z'],z0)
        elif 'y' in kw:
            self.s = y2s(kw['y'],z0)
        else:
            s=zeros(len(frequency))

        self.frequency = frequency
        self.z0 = z0
        self.name = name
        self.comments = comments

    @classmethod
    def from_ntwkv1( cls, network):
        return cls(frequency = network.frequency,
                   s = network.s,
                   z0 = network.z0,
                   name = network.name,
                   comments = network.comments,
                   )
    def __str__(self):
        f = self.frequency
        if self.name is None:
            name = ''
        else:
            name = self.name

        if len(npy.shape(self.z0)) == 0:
            z0 = str(self.z0)
        else:
            z0 = str(self.z0[0,:])

        output = '%i-Port Network: \'%s\',  %s, z0=%s' % (self.nports, name, str(f), z0)

        return output

    def __repr__(self):
        return self.__str__()

    def __call__(self, i,j):
        n = self.copy()
        n.s = n.s[:,i,j]
        return n

    def __len__(self):
        '''
        length of frequency axis
        '''
        return len(self.frequency)

    def __getitem__(self,key):
        '''
        Slices a Network object based on an index, or human readable string

        Parameters
        -----------
        key : str, or int
            if int; then it is interpreted as the index of the frequency
            if str, then should be like '50.1-75.5ghz', or just '50'.
            If the frequency unit is omitted then self.frequency.unit is
            used.

        Examples
        -----------
        >>> from skrf.data import ring_slot
        >>> a = ring_slot['80-90ghz']
        >>> a.plot_s_db()
        '''

        if isinstance(key, str):
            # they passed a string. try to read the string and convert
            # it into a  slice. then slice self on that
            re_numbers = re.compile('.*\d')
            re_hyphen = re.compile('\s*-\s*')
            re_letters = re.compile('[a-zA-Z]+')

            freq_unit = re.findall(re_letters,key)

            if len(freq_unit) == 0:
                freq_unit = self.frequency.unit
            else:
                freq_unit = freq_unit[0]

            key_nounit = re.sub(re_letters,'',key)
            edges  = re.split(re_hyphen,key_nounit)

            edges_freq = Frequency.from_f([float(k) for k in edges],
                                        unit = freq_unit)
            if len(edges_freq) ==2:
                slicer=slice_domain(self.frequency.f, edges_freq.f)
            elif len(edges_freq)==1:
                key = find_nearest_index(self.frequency.f, edges_freq.f[0])
                slicer = slice(key,key+1,1)
            else:
                raise ValueError()

            key = slicer

        try:

            output = self.copy()
            output.frequency.f = npy.array(output.frequency.f[key]).reshape(-1)
            output.z0 = output.z0[key,:]
            output.s = output.s[key,:,:]
            return output

        except(IndexError):
            raise IndexError('slicing frequency/index is incorrect')



    def copy(self):
        ntwk = Network(frequency =self.frequency.copy(),
                       s = self.s.val.copy(),
                       z0 = self.z0.copy(),
                       name = self.name,
                       comments = self.comments,
                       )

        return ntwk

    @property
    def s(self):
        '''
        Scattering Parameters
        '''
        return self._s

    @s.setter
    def s(self,s):
        self._s = S(self, s)

    @property
    def z(self):
        return Z(self)

    @z.setter
    def z(self,z):
        self.s = z2s(z,self.z0)

    @property
    def y(self):
        return Y(self)

    @y.setter
    def y(self,y):
        self.s = y2s(y,self.z0)

    @property
    def t(self):
        return T(self)

    @t.setter
    def t(self,t):
        raise NotImplementedError()

    @property
    def s_time(self):
        return STime(self)

    @s_time.setter
    def s_time(self):
        raise NotImplementedError()





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

    @property
    def z0(self):
        '''
        The port impedance
        '''
        return self._z0

    @z0.setter
    def z0(self,z0):
        self._z0 = fix_z0_shape(z0, len(self.frequency),nports=self.nports)

    @property
    def port_tuples(self):
        '''
        Returns a list of tuples, for each port index pair

        A convenience function for the common task fo iterating over
        all s-parameters index pairs

        This just calls:
        `[(y,x) for x in range(self.nports) for y in range(self.nports)]`
        '''
        return [(y,x) for x in range(self.nports) for y in range(self.nports)]

    def windowed(self, window=('kaiser',6),  normalize = True):
        '''
        Return a windowed version of s-matrix. Used in time-domain analysis.

        When using time domain through :attr:`s_time_db`,
        or similar properties, the spectrum is ussually windowed,
        before the IFFT is taken. This is done to
        compensate for the band-pass nature of a spectrum [1]_ .

        This function calls :func:`scipy.signal.get_window` which gives
        more details about the windowing.

        Parameters
        -----------
        window : string, float, or tuple
            The type of window to create. See :func:`scipy.signal.get_window`
            for details.
        normalize : bool
            Normalize the window to preserve power. ie
            sum(ntwk.s,axis=0) == sum(ntwk.windowed().s,axis=0)

        Examples
        -----------
        >>> ntwk = rf.Network('myfile.s2p')
        >>> ntwk_w = ntwk.windowed()
        >>> ntwk_w.plot_s_time_db()

        References
        -------------
        .. [1] Agilent Time Domain Analysis Using a Network Analyzer Application Note 1287-12

        '''

        windowed = self.copy()
        window = signal.get_window(window, len(self))
        window =window.reshape(-1,1,1) * npy.ones((len(self),
                                                   self.nports,
                                                   self.nports))
        windowed.s  = windowed.s[:] * window
        if normalize:
            # normalize the s-parameters to account for power lost in windowing
            windowed.s = windowed.s[:] * npy.sum(self.s.mag[:],axis=0)/\
                npy.sum(windowed.s.mag[:],axis=0)

        return windowed



def fix_z0_shape( z0, nfreqs, nports):
    '''
    Make a port impedance of correct shape for a given network's matrix

    This attempts to broadcast z0 to satisfy
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

def fix_parameter_shape(s):
    s_shape= npy.shape(s)
    if len(s_shape) <3:
        if len(s_shape) == 2:
            # reshape to kx1x1, this simplifies indexing in function
            s = npy.reshape(s,(-1,s_shape[0],s_shape[0]))
        else:
            s = npy.reshape(s,(-1,1,1))
    return s

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
    s = s.copy() # to prevent the original array from being altered
    s = fix_parameter_shape(s)
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)

    z = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))

    s[s==1.] = 1. + 1e-12 # solve numerical singularity
    s[s==-1.] = -1. + 1e-12 # solve numerical singularity
    for fidx in xrange(s.shape[0]):
        sqrtz0 = npy.mat(npy.sqrt(npy.diagflat(z0[fidx])))
        z[fidx] = sqrtz0 * (I-s[fidx])**-1 * (I+s[fidx]) * sqrtz0
    return z

def s2y(s,z0=50):
    '''
    convert scattering parameters [#]_ to admittance parameters [#]_


    .. math::
        y = \\sqrt {y_0} \\cdot (I - s)(I + s)^{-1} \\cdot \\sqrt{y_0}

    Parameters
    ------------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number
        port impedances

    Returns
    ---------
    y : complex array-like
        admittance parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    '''
    s = s.copy() # to prevent the original array from being altered
    s = fix_parameter_shape(s)
    nfreqs, nports, nports = s.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    y = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    s[s==-1.] = -1. + 1e-12 # solve numerical singularity
    s[s==1.] = 1. + 1e-12 # solve numerical singularity
    for fidx in xrange(s.shape[0]):
        sqrty0 = npy.mat(npy.sqrt(npy.diagflat(1.0/z0[fidx])))
        y[fidx] = sqrty0*(I-s[fidx])*(I+s[fidx])**-1*sqrty0
    return y

def s2t(s):
    '''
    Converts scattering parameters [#]_ to scattering transfer parameters [#]_ .

    transfer parameters are also referred to as
    'wave cascading matrix', this function only operates on 2-port
    networks.

    Parameters
    -----------
    s : :class:`numpy.ndarray` (shape fx2x2)
        scattering parameter matrix

    Returns
    -------
    t : numpy.ndarray
        scattering transfer parameters (aka wave cascading matrix)

    See Also
    ---------
    inv : calculates inverse s-parameters

    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    -----------
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    '''
    #TODO: check rank(s) ==2
    s = s.copy() # to prevent the original array from being altered
    s = fix_parameter_shape(s)
    t = npy.array([
        [-1*(s[:,0,0]*s[:,1,1]- s[:,1,0]*s[:,0,1])/s[:,1,0],
            -s[:,1,1]/s[:,1,0]],
        [s[:,0,0]/s[:,1,0],
            1/s[:,1,0] ]
        ]).transpose()
    return t

def s2time(s,window =('kaiser',6),  normalize = True):
    '''
    '''
    s = s.copy() # to prevent the original array from being altered
    s = fix_parameter_shape(s)
    nfreqs, nports, nports = s.shape


    window = signal.get_window(window,nfreqs)
    window =window.reshape(-1,1,1) * npy.ones(s.shape)
    windowed = s * window
    if normalize:
        # normalize the s-parameters to account for power lost in windowing
        norm_factor = npy.sum(abs(s),axis=0)/\
                      npy.sum(abs(windowed),axis=0)
        windowed = windowed*norm_factor

    time = fft.ifftshift(fft.ifft(windowed, axis=0), axes=0)
    return time


def z2s(z, z0=50):
    '''
    convert impedance parameters [1]_ to scattering parameters [2]_

    .. math::
        s = (\\sqrt{y_0} \\cdot z \\cdot \\sqrt{y_0} - I)(\\sqrt{y_0} \\cdot z \\cdot\\sqrt{y_0} + I)^{-1}

    Parameters
    ------------
    z : complex array-like
        impedance parameters
    z0 : complex array-like or number
        port impedances

    Returns
    ---------
    s : complex array-like
        scattering parameters



    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/impedance_parameters
    .. [2] http://en.wikipedia.org/wiki/S-parameters
    '''
    z = z.copy() # to prevent the original array from being altered
    z = fix_parameter_shape(z)
    nfreqs, nports, nports = z.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    s = npy.zeros(z.shape, dtype='complex')
    I = npy.mat(npy.identity(z.shape[1]))
    for fidx in xrange(z.shape[0]):
        sqrty0 = npy.mat(npy.sqrt(npy.diagflat(1.0/z0[fidx])))
        s[fidx] = (sqrty0*z[fidx]*sqrty0 - I) * (sqrty0*z[fidx]*sqrty0 + I)**-1
    return s

def z2y(z):
    '''
    convert impedance parameters [#]_ to admittance parameters [#]_


    .. math::
        y = z^{-1}

    Parameters
    ------------
    z : complex array-like
        impedance parameters

    Returns
    ---------
    y : complex array-like
        admittance parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    '''
    z = z.copy() # to prevent the original array from being altered
    z = fix_parameter_shape(z)
    return npy.array([npy.mat(z[f,:,:])**-1 for f in xrange(z.shape[0])])

def z2t(z):
    '''
    Not Implemented yet

    convert impedance parameters [#]_ to scattering transfer parameters [#]_


    Parameters
    ------------
    z : complex array-like or number
        impedance parameters

    Returns
    ---------
    s : complex array-like or number
        scattering parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t


    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    '''
    raise (NotImplementedError)

def y2s(y, z0=50):
    '''
    convert admittance parameters [#]_ to scattering parameters [#]_


    .. math::
        s = (I - \\sqrt{z_0} \\cdot y \\cdot \\sqrt{z_0})(I + \\sqrt{z_0} \\cdot y \\cdot \\sqrt{z_0})^{-1}

    Parameters
    ------------
    y : complex array-like
        admittance parameters

    z0 : complex array-like or number
        port impedances

    Returns
    ---------
    s : complex array-like or number
        scattering parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t


    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    '''
    y = y.copy() # to prevent the original array from being altered
    y = fix_parameter_shape(y)
    nfreqs, nports, nports = y.shape
    z0 = fix_z0_shape(z0, nfreqs, nports)
    s = npy.zeros(y.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    for fidx in xrange(s.shape[0]):
        sqrtz0 = npy.mat(npy.sqrt(npy.diagflat(z0[fidx])))
        s[fidx] = (I - sqrtz0*y[fidx]*sqrtz0) * (I + sqrtz0*y[fidx]*sqrtz0)**-1
    return s

def y2z(y):
    '''
    convert admittance parameters [#]_ to impedance parameters [#]_


    .. math::
        z = y^{-1}

    Parameters
    ------------
    y : complex array-like
        admittance parameters

    Returns
    ---------
    z : complex array-like
        impedance parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    '''
    y = y.copy() # to prevent the original array from being altered
    y = fix_parameter_shape(y)
    return npy.array([npy.mat(y[f,:,:])**-1 for f in xrange(y.shape[0])])

def y2t(y):
    '''
    Not Implemented Yet

    convert admittance parameters [#]_ to scattering-transfer parameters [#]_


    Parameters
    ------------
    y : complex array-like or number
        impedance parameters

    Returns
    ---------
    t : complex array-like or number
        scattering parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    '''
    raise (NotImplementedError)

def t2s(t):
    '''
    converts scattering transfer parameters [#]_ to scattering parameters [#]_

    transfer parameters are also refered to as
    'wave cascading matrix', this function only operates on 2-port
    networks. this function only operates on 2-port scattering
    parameters.

    Parameters
    -----------
    t : :class:`numpy.ndarray` (shape fx2x2)
            scattering transfer parameters

    Returns
    -------
    s : :class:`numpy.ndarray`
            scattering parameter matrix.

    See Also
    ---------
    inv : calculates inverse s-parameters
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    -----------
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    '''
    #TODO: check rank(s) ==2
    t = t.copy() # to prevent the original array from being altered
    t = fix_parameter_shape(t)
    s = npy.array([
        [t[:,0,1]/t[:,1,1],
             1/t[:,1,1]],
        [(t[:,0,0]*t[:,1,1]- t[:,1,0]*t[:,0,1])/t[:,1,1],
            -1*t[:,1,0]/t[:,1,1] ]
        ]).transpose()
    return s

def t2z(t):
    '''
    Not Implemented  Yet

    Convert scattering transfer parameters [#]_ to impedance parameters [#]_



    Parameters
    ------------
    t : complex array-like or number
        impedance parameters

    Returns
    ---------
    z : complex array-like or number
        scattering parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    '''
    raise (NotImplementedError)

def t2y(t):
    '''
    Not Implemented Yet

    Convert scattering transfer parameters to admittance parameters [#]_




    Parameters
    ------------
    t : complex array-like or number
        t-parameters

    Returns
    ---------
    y : complex array-like or number
        admittance parameters

    See Also
    ----------
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2z
    t2s
    t2z
    t2y
    Network.s
    Network.y
    Network.z
    Network.t

    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters

    '''
    raise (NotImplementedError)
