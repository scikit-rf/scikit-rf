
#       network.py
#
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       Copyright 2010 lihan chen
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

'''
.. module:: skrf.network
========================================
network (:mod:`skrf.network`)
========================================


Provides a n-port network class and associated functions.

Most of the functionality in this module is provided as methods and
properties of the :class:`Network` Class.


Network Class
===============

.. autosummary::
    :toctree: generated/
    
    Network


Connecting Networks
===============================

.. autosummary::
    :toctree: generated/
    
    connect
    innerconnect
    cascade
    de_embed
    flip


Interpolation and Stitching
=============================

.. autosummary::
    :toctree: generated/
    
    Network.resample
    Network.interpolate
    Network.interpolate_self
    Network.interpolate_from_f
    stitch

IO
====

.. autosummary::
    
    skrf.io.general.read
    skrf.io.general.write
    Network.write
    Network.write_touchstone
    Network.read
    
Noise
============
.. autosummary::
    :toctree: generated/
    
    Network.add_noise_polar
    Network.add_noise_polar_flatband
    Network.multiply_noise

    
Supporting Functions
======================

.. autosummary::
    :toctree: generated/
    
    inv
    connect_s
    innerconnect_s
    s2z
    s2y
    s2t
    z2s
    z2y
    z2t
    y2s
    y2z
    y2t
    t2s
    t2z
    t2y


Misc Functions
=====================
.. autosummary::
    :toctree: generated/

    average
    Network.nudge


'''
import os
import warnings
import cPickle as pickle    
from cPickle import UnpicklingError
from copy import deepcopy as copy


import numpy as npy
import pylab as plb
from scipy import stats         # for Network.add_noise_*
from scipy.interpolate import interp1d # for Network.interpolate()
import unittest # fotr unitest.skip 
import  mathFunctions as mf

from frequency import Frequency
from plotting import *#smith, plot_rectangular, plot_smith, plot_complex_polar
from tlineFunctions import zl_2_Gamma0
from util import get_fid
## later imports. delayed to solve circular dependencies
#from io.general import read, write
#from io import touchstone





class Network(object):
    '''

    A n-port electrical network [#]_.
    
    For instructions on how to create Network see  :func:`__init__`.
    
    A n-port network may be defined by three quantities,
     * network parameter matrix (s, z, or y-matrix)
     * port characteristic impedance matrix
     * frequency information

    The :class:`Network` class stores these data structures internally
    in the form of complex :class:`numpy.ndarray`'s. These arrays are not
    interfaced directly but instead through the use of the properties:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s`              scattering parameter matrix
    :attr:`z0`             characteristic impedance matrix
    :attr:`f`              frequency vector
    =====================  =============================================

    Although these docs focus on s-parameters, other equivalent network 
    representations such as :attr:`z` and  :attr:`y` are 
    available. Scalar projections of the complex network parameters 
    are accesable through properties as well. These also return 
    :class:`numpy.ndarray`'s.

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s_re`           real part of the s-matrix
    :attr:`s_im`           imaginary part of the s-matrix
    :attr:`s_mag`          magnitude of the s-matrix
    :attr:`s_db`           magnitude in log scale of the s-matrix
    :attr:`s_deg`          phase of the s-matrix in degrees
    =====================  =============================================

    The following operations act on the networks s-matrix. 

    =====================  =============================================
    Operator               Function
    =====================  =============================================
    \+                     element-wise addition of the s-matrix
    \-                     element-wise difference of the s-matrix
    \*                     element-wise multiplication of the s-matrix
    \/                     element-wise division of the s-matrix
    \*\*                   cascading (only for 2-ports)
    \//                    de-embedding (for 2-ports, see :attr:`inv`)
    =====================  =============================================

    Different components of the :class:`Network` can be visualized
    through various plotting methods. These methods can be used to plot
    individual elements of the s-matrix or all at once. For more info
    about plotting see the :doc:`../../tutorials/plotting` tutorial.

    =========================  =============================================
    Method                     Meaning
    =========================  =============================================
    :func:`plot_s_smith`       plot complex s-parameters on smith chart
    :func:`plot_s_re`          plot real part of s-parameters vs frequency
    :func:`plot_s_im`          plot imaginary part of s-parameters vs frequency
    :func:`plot_s_mag`         plot magnitude of s-parameters vs frequency
    :func:`plot_s_db`          plot magnitude (in dB) of s-parameters vs frequency
    :func:`plot_s_deg`         plot phase of s-parameters (in degrees) vs frequency
    :func:`plot_s_deg_unwrap`  plot phase of s-parameters (in unwrapped degrees) vs frequency
    =========================  =============================================

    :class:`Network`  objects can be  created from a touchstone or pickle
    file  (see :func:`__init__`), by a 
    :class:`~skrf.media.media.Media` object, or manually by assigning the 
    network properties directly. :class:`Network`  objects
    can be saved to disk in the form of touchstone files with the
    :func:`write_touchstone` method.

    An exhaustive list of :class:`Network` Methods and Properties
    (Attributes) are given below
    
    References
    ------------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
    '''
    # used for testing s-parameter equivalence
    global ALMOST_ZERO
    ALMOST_ZERO = 1e-6

    global PRIMARY_PROPERTIES
    PRIMARY_PROPERTIES = [ 's','z','y','a']
    
    global COMPONENT_FUNC_DICT
    COMPONENT_FUNC_DICT = {
        're'    : npy.real,
        'im'    : npy.imag,
        'mag'   : npy.abs,
        'db'    : mf.complex_2_db,
        'rad'   : npy.angle,
        'deg'   : lambda x: npy.angle(x, deg=True),
        'arcl'  : lambda x: npy.angle(x) * npy.abs(x),
        'rad_unwrap'    : lambda x: mf.unwrap_rad(npy.angle(x)),
        'deg_unwrap'    : lambda x: mf.radian_2_degree(mf.unwrap_rad(\
            npy.angle(x))),
        'arcl_unwrap'   : lambda x: mf.unwrap_rad(npy.angle(x)) *\
            npy.abs(x),
        }
    # provides y-axis labels to the plotting functions
    global Y_LABEL_DICT
    Y_LABEL_DICT = {
        're'    : 'Real Part',
        'im'    : 'Imag Part',
        'mag'   : 'Magnitude',
        'abs'   : 'Magnitude',
        'db'    : 'Magnitude (dB)',
        'deg'   : 'Phase (deg)',
        'deg_unwrap'    : 'Phase (deg)',
        'rad'   : 'Phase (rad)',
        'rad_unwrap'    : 'Phase (rad)',
        'arcl'  : 'Arc Length',
        'arcl_unwrap'   : 'Arc Length',
        }

        
    ## CONSTRUCTOR
    def __init__(self, file = None, name = None , comments = None, **kwargs):
        '''
        Network constructor.

        Creates an n-port microwave network from a `file` or directly 
        from data. If no file or data is given, then an empty Network 
        is created.
            
        Parameters
        ------------

        file : str or file-object
            file to load information from. supported formats are:
             * touchstone file (.s?p)
             * pickled Network (.ntwk, .p) see :func:`write`
        name : str
            Name of this Network. if None will try to use file, if 
            its a str
        comments : str
            Comments associated with the Network 
        \*\*kwargs : 
            key word arguments can be used to assign properties of the 
            Network, such as `s`, `f` and `z0`. 
            
        Examples
        ------------
        From a touchstone
        
        >>> n = rf.Network('ntwk1.s2p')
        
        From a pickle file
        
        >>> n = rf.Network('ntwk1.ntwk')
        
        Create a blank network, then fill in values
        
        >>> n = rf.Network() 
        >>> n.f, n.s, n.z0 = [1,2,3],[1,2,3], [1,2,3]
        
        Directly from values
        
        >>> n = rf.Network(f=[1,2,3],s=[1,2,3],z0=[1,2,3])
        
        See Also
        -----------
        read : read a network from a file
        write : write a network to a file, using pickle
        write_touchstone : write a network to a touchstone file
        '''
        
        # allow for old kwarg for backward compatability
        if kwargs.has_key('touchstone_filename'):
            file = kwargs['touchstone_filename']
        
        
        if file is not None:
            # allows user to pass filename or file obj
            # open file in 'binary' mode because we are going to try and 
            # upickle it first
            fid = get_fid(file,'rb') 
            
            try: 
                self.read(fid)
            except(UnpicklingError):
                # if unpickling doesnt work then, close fid, reopen in 
                # non-binary mode and try to read it as touchstone
                fid.close()
                fid = get_fid(file)
                self.read_touchstone(fid)
            
            if name is None and isinstance(file,basestring):
                name = os.path.splitext(os.path.basename(file))[0]
        
        self.name = name
        self.comments = comments
        
        # allow properties to be set through the constructor 
        for attr in PRIMARY_PROPERTIES + ['frequency','z0','f']:
            if kwargs.has_key(attr):
                self.__setattr__(attr,kwargs[attr])

        
        
        #self.nports = self.number_of_ports
        self.__generate_plot_functions()

    ## OPERATORS
    def __pow__(self,other):
        '''
        cascade this network with another network

        port 1 of this network is connected to port 0 or the other
        network
        '''
        return connect(self,1,other,0)

    def __floordiv__(self,other):
        '''
        de-embeding another network[s], from this network

        See Also
        ----------
        inv : inverse s-parameters
        '''
        try:
            # if they passed 1 ntwks and a tuple of ntwks,
            # then deEmbed like A.inv*C*B.inv
            b = other[0]
            c = other[1]
            result =  copy (self)
            result.s =  (b.inv**self**c.inv).s
            #flip(de_embed( flip(de_embed(c.s,self.s)),b.s))
            return result
        except TypeError:
            pass

        if other.number_of_ports == 2:
            result = self.copy()
            result.s = (other.inv**self).s
            #de_embed(self.s,other.s)
            return result
        else:
            raise IndexError('Incorrect number of ports.')

    def __mul__(self,other):
        '''
        Element-wise complex multiplication of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s * other.s
        else:
            # other may be an array or a number
            result.s = self.s * npy.array(other).reshape(-1,1,1)
            
        return result
    
    def __rmul__(self,other):
        '''
        Element-wise complex multiplication of s-matrix
        '''
        
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s * other.s
        else:
            # other may be an array or a number
            result.s = self.s * npy.array(other).reshape(-1,1,1)
            
        return result
    
    def __add__(self,other):
        '''
        Element-wise complex addition of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s + other.s
        else:
            # other may be an array or a number
            result.s = self.s + npy.array(other).reshape(-1,1,1)
            
        return result
    
    def __radd__(self,other):
        '''
        Element-wise complex addition of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s + other.s
        else:
            # other may be an array or a number
            result.s = self.s + npy.array(other).reshape(-1,1,1)
            
        return result
    

    def __sub__(self,other):
        '''
        Element-wise complex subtraction of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s - other.s
        else:
            # other may be an array or a number
            result.s = self.s - npy.array(other).reshape(-1,1,1)
            
        return result
    
    def __rsub__(self,other):
        '''
        Element-wise complex subtraction of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = other.s - self.s
        else:
            # other may be an array or a number
            result.s = npy.array(other).reshape(-1,1,1) - self.s
            
        return result

    def __div__(self,other):
        '''
        Element-wise complex multiplication of s-matrix
        '''
        result = self.copy()
        
        if isinstance(other, Network):
            self.__compatable_for_scalar_operation_test(other)
            result.s = self.s / other.s
        else:
            # other may be an array or a number
            result.s = self.s / npy.array(other).reshape(-1,1,1)
            
        return result
    

    def __eq__(self,other):
        if npy.all(npy.abs(self.s - other.s) < ALMOST_ZERO):
            return True
        else:
            return False

    def __ne__(self,other):
        return (not self.__eq__(other))

    def __getitem__(self,key):
        '''
        returns a Network object at a given single frequency
        '''
        a = self.z0# HACK: to force getter for z0 to re-shape it
        output = self.copy()
        output.s = output.s[key,:,:]
        output.z0 = output.z0[key,:]
        output.frequency.f = npy.array(output.frequency.f[key]).reshape(-1)

        return output

    def __str__(self):
        '''
        '''
        f = self.frequency
        if self.name is None:
            name = ''
        else:
            name = self.name

        if len(npy.shape(self.z0)) == 0:
            z0 = str(self.z0)
        else:
            z0 = str(self.z0[0,:])

        output = '%i-Port Network: \'%s\',  %s, z0=%s' % (self.number_of_ports, name, str(f), z0)

        return output

    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        '''
        length of frequency axis
        '''
        return len(self.s)
    
    
       
    ## INTERNAL CODE GENERATION METHODS
    def __compatable_for_scalar_operation_test(self, other):
        '''
        tests to make sure other network's s-matrix is of same shape
        '''
        if other.frequency  != self.frequency:
            raise IndexError('Networks must have same frequency. See `Network.interpolate`')

        if other.s.shape != self.s.shape:
            raise IndexError('Networks must have same number of ports.')
    
    def __generate_secondary_properties(self):
        '''
        creates numerous `secondary properties` which are various
        different scalar projects of the promary properties. the primary
        properties are  s,z, and y.
        '''
        for prop_name in PRIMARY_PROPERTIES:
            for func_name in COMPONENT_FUNC_DICT:
                func = COMPONENT_FUNC_DICT[func_name]
                def fget(self, f=func, p = prop_name):
                    return f(getattr(self,p))
                
                doc = '''
                The %s component of the %s-matrix
                
                
                See Also
                ----------
                %s
                '''%(func_name, prop_name, prop_name)
                
                setattr(self.__class__,'%s_%s'%(prop_name, func_name),\
                    property(fget, doc = doc))

    def __generate_plot_functions(self):
        '''
        '''
        for prop_name in PRIMARY_PROPERTIES:

            def plot_prop_polar(self, 
                m=None, n=None, ax=None,
                show_legend=True ,prop_name=prop_name,*args, **kwargs):

                # create index lists, if not provided by user
                if m is None:
                    M = range(self.number_of_ports)
                else:
                    M = [m]
                if n is None:
                    N = range(self.number_of_ports)
                else:
                    N = [n]

                if 'label'  not in kwargs.keys():
                    gen_label = True
                else:
                    gen_label = False

                
                was_interactive = plb.isinteractive
                if was_interactive:
                    plb.interactive(False)
                    
                for m in M:
                    for n in N:
                        # set the legend label for this trace to the networks
                        # name if it exists, and they didnt pass a name key in
                        # the kwargs
                        if gen_label:
                            if self.name is None:
                                if plb.rcParams['text.usetex']:
                                    label_string = '$%s_{%i%i}$'%\
                                    (prop_name[0].upper(),m+1,n+1)
                                else:
                                    label_string = '%s%i%i'%\
                                    (prop_name[0].upper(),m+1,n+1)
                            else:
                                if plb.rcParams['text.usetex']:
                                    label_string = self.name+', $%s_{%i%i}$'%\
                                    (prop_name[0].upper(),m+1,n+1)
                                else:
                                    label_string = self.name+', %s%i%i'%\
                                    (prop_name[0].upper(),m+1,n+1)
                            kwargs['label'] = label_string
        
                        # plot the desired attribute vs frequency
                        plot_complex_polar(
                            z = getattr(self,prop_name)[:,m,n],
                            *args, **kwargs)

                if was_interactive:
                    plb.interactive(True)
                    plb.draw()
                    plb.show()
            
            plot_prop_polar.__doc__ = '''
    plot the Network attribute :attr:`%s` vs frequency.
    
    Parameters
    -----------
    m : int, optional
        first index of s-parameter matrix, if None will use all 
    n : int, optional
        secon index of the s-parameter matrix, if None will use all  
    ax : :class:`matplotlib.Axes` object, optional
        An existing Axes object to plot on
    show_legend : Boolean
        draw legend or not
    attribute : string
        Network attribute to plot 
    y_label : string, optional
        the y-axis label
    
    \*args,\\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot` 
    
    Notes
    -------
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    ------------
    >>> myntwk.plot_%s(m=1,n=0,color='r')
    '''%(prop_name,prop_name)

            setattr(self.__class__,'plot_%s_polar'%(prop_name), \
                plot_prop_polar)

            def plot_prop_rect(self, 
                m=None, n=None, ax=None,
                show_legend=True,prop_name=prop_name,*args, **kwargs):

                # create index lists, if not provided by user
                if m is None:
                    M = range(self.number_of_ports)
                else:
                    M = [m]
                if n is None:
                    N = range(self.number_of_ports)
                else:
                    N = [n]

                if 'label'  not in kwargs.keys():
                    gen_label = True
                else:
                    gen_label = False

                
                #was_interactive = plb.isinteractive
                #if was_interactive:
                #    plb.interactive(False)
                    
                for m in M:
                    for n in N:
                        # set the legend label for this trace to the networks
                        # name if it exists, and they didnt pass a name key in
                        # the kwargs
                        if gen_label:
                            if self.name is None:
                                if plb.rcParams['text.usetex']:
                                    label_string = '$%s_{%i%i}$'%\
                                    (prop_name[0].upper(),m+1,n+1)
                                else:
                                    label_string = '%s%i%i'%\
                                    (prop_name[0].upper(),m+1,n+1)
                            else:
                                if plb.rcParams['text.usetex']:
                                    label_string = self.name+', $%s_{%i%i}$'%\
                                    (prop_name[0].upper(),m+1,n+1)
                                else:
                                    label_string = self.name+', %s%i%i'%\
                                    (prop_name[0].upper(),m+1,n+1)
                            kwargs['label'] = label_string
        
                        # plot the desired attribute vs frequency
                        plot_complex_rectangular(
                            z = getattr(self,prop_name)[:,m,n],
                            *args, **kwargs)

                #if was_interactive:
                #    plb.interactive(True)
                #    plb.draw()
                #    plb.show()
            
            plot_prop_rect.__doc__ = '''
    plot the Network attribute :attr:`%s` vs frequency.
    
    Parameters
    -----------
    m : int, optional
        first index of s-parameter matrix, if None will use all 
    n : int, optional
        secon index of the s-parameter matrix, if None will use all  
    ax : :class:`matplotlib.Axes` object, optional
        An existing Axes object to plot on
    show_legend : Boolean
        draw legend or not
    attribute : string
        Network attribute to plot 
    y_label : string, optional
        the y-axis label
    
    \*args,\\**kwargs : arguments, keyword arguments
        passed to :func:`matplotlib.plot` 
    
    Notes
    -------
    This function is dynamically generated upon Network
    initialization. This is accomplished by calling
    :func:`plot_vs_frequency_generic`

    Examples
    ------------
    >>> myntwk.plot_%s(m=1,n=0,color='r')
    '''%(prop_name,prop_name)

            setattr(self.__class__,'plot_%s_complex'%(prop_name), \
                plot_prop_rect)


            for func_name in COMPONENT_FUNC_DICT:
                attribute = '%s_%s'%(prop_name, func_name)
                y_label = Y_LABEL_DICT[func_name]
                
                def plot_func(self,  m=None, n=None, ax=None,
                    show_legend=True,attribute=attribute,
                    y_label=y_label,*args, **kwargs):

                    # create index lists, if not provided by user
                    if m is None:
                        M = range(self.number_of_ports)
                    else:
                        M = [m]
                    if n is None:
                        N = range(self.number_of_ports)
                    else:
                        N = [n]

                    if 'label'  not in kwargs.keys():
                        gen_label = True
                    else:
                        gen_label = False

                    #TODO: turn off interactive plotting for performance
                    # this didnt work because it required a show()
                    # to be called, which in turn, disrupted testCases
                    #
                    #was_interactive = plb.isinteractive
                    #if was_interactive:
                    #    plb.interactive(False)
                        
                    for m in M:
                        for n in N:
                            # set the legend label for this trace to the networks
                            # name if it exists, and they didnt pass a name key in
                            # the kwargs
                            if gen_label:
                                if self.name is None:
                                    if plb.rcParams['text.usetex']:
                                        label_string = '$%s_{%i%i}$'%\
                                        (attribute[0].upper(),m+1,n+1)
                                    else:
                                        label_string = '%s%i%i'%\
                                        (attribute[0].upper(),m+1,n+1)
                                else:
                                    if plb.rcParams['text.usetex']:
                                        label_string = self.name+', $%s_{%i%i}$'%\
                                        (attribute[0].upper(),m+1,n+1)
                                    else:
                                        label_string = self.name+', %s%i%i'%\
                                        (attribute[0].upper(),m+1,n+1)
                                kwargs['label'] = label_string
            
                            # plot the desired attribute vs frequency
                            plot_rectangular(
                                x = self.frequency.f_scaled,
                                y = getattr(self,attribute)[:,m,n],
                                x_label = 'Frequency (' + \
                                    self.frequency.unit +')',
                                y_label = y_label,
                                *args, **kwargs)

                    #if was_interactive:
                    #    plb.interactive(True)
                    #    plb.draw()
                    #    #plb.show()
                
                plot_func.__doc__ = '''
        plot the Network attribute :attr:`%s` vs frequency.
        
        Parameters
        -----------
        m : int, optional
            first index of s-parameter matrix, if None will use all 
        n : int, optional
            secon index of the s-parameter matrix, if None will use all  
        ax : :class:`matplotlib.Axes` object, optional
            An existing Axes object to plot on
        show_legend : Boolean
            draw legend or not
        attribute : string
            Network attribute to plot 
        y_label : string, optional
            the y-axis label
        
        \*args,\\**kwargs : arguments, keyword arguments
            passed to :func:`matplotlib.plot` 
        
        Notes
        -------
        This function is dynamically generated upon Network
        initialization. This is accomplished by calling
        :func:`plot_vs_frequency_generic`

        Examples
        ------------
        >>> myntwk.plot_%s(m=1,n=0,color='r')
        '''%(attribute,attribute)

                setattr(self.__class__,'plot_%s'%(attribute), \
                    plot_func)

    def __generate_subnetworks(self):
        '''
        generates all one-port sub-networks
        '''
        for m in range(self.number_of_ports):
            for n in range(self.number_of_ports):
                def fget(self,m=m,n=n):
                    ntwk = self.copy()
                    ntwk.s = self.s[:,m,n]
                    ntwk.z0 = self.z0[:,m]
                    return ntwk
                doc = '''
                one-port sub-network.
                '''
                setattr(self.__class__,'s%i%i'%(m+1,n+1),\
                    property(fget,doc=doc))

    ## PRIMARY PROPERTIES
    @property
    def s(self):
        '''
        Scattering parameter matrix.

        The s-matrix[#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so s11 can be accessed by 
        taking the slice s[:,0,0].  
        

        Returns
        ---------
        s : complex :class:`numpy.ndarray` of shape `fxnxn`
                the scattering parameter matrix.
        
        See Also
        ------------
        s 
        y
        z
        t
        a
        
        References
        ------------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters
        '''
        return self._s

    @s.setter
    def s(self, s):
        '''
        the input s-matrix should be of shape fxnxn,
        where f is frequency axis and n is number of ports
        '''
        s_shape= npy.shape(s)
        if len(s_shape) <3:
            if len(s_shape) == 2:
                # reshape to kx1x1, this simplifies indexing in function
                s = npy.reshape(s,(-1,s_shape[0],s_shape[0]))
            else:
                s = npy.reshape(s,(-1,1,1))

        self._s = npy.array(s,dtype=complex)
        self.__generate_secondary_properties()
        self.__generate_subnetworks()
       
    @property
    def y(self):
        '''
        Admittance parameter matrix.

        The y-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so y11 can be accessed by 
        taking the slice `y[:,0,0]`.  
        

        Returns
        ---------
        y : complex :class:`numpy.ndarray` of shape `fxnxn`
                the admittance parameter matrix.

        See Also
        ------------
        s 
        y
        z
        t
        a

        References
        ------------
        .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
        '''
        return s2y(self._s, self.z0)

    @y.setter
    def y(self, value):
        self._s = y2s(value, self.z0)
    
    @property
    def z(self):
        '''
        Impedance parameter matrix.

        The z-matrix  [#]_ is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so z11 can be accessed by 
        taking the slice `z[:,0,0]`.  
        

        Returns
        ---------
        z : complex :class:`numpy.ndarray` of shape `fxnxn`
                the Impedance parameter matrix.
                
        See Also
        ------------
        s 
        y
        z
        t
        a

        References
        ------------
        .. [#] http://en.wikipedia.org/wiki/impedance_parameters
        '''
        return s2z(self._s, self.z0)
    
    @z.setter
    def z(self, value):
        self._s = z2s(value, self.z0)
    
    @property
    def t(self):
        '''
        Scattering transfer parameters

        The t-matrix [#]_ is a 3-dimensional :class:`numpy.ndarray` 
        which has shape `fx2x2`, where `f` is frequency axis.
        Note that indexing starts at 0, so t11 can be accessed by 
        taking the slice `t[:,0,0]`.  
        
        The t-matrix, also known as the wave cascading matrix, is 
        only defined for a 2-port Network.

        Returns
        --------
        t : complex numpy.ndarry of shape `fx2x2`
                t-parameters, aka scattering transfer parameters

        
        See Also
        ------------
        s 
        y
        z
        t
        a
        
        References
        -----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters
        '''
        return s2t(self.s)
    
    @property 
    def a(self):
        '''
        Active scattering parameter matrix.
        
        Active scattering parameters are simply inverted s-parameters, 
        defined as a = 1/s. Useful in analysis of active networks.
        The a-matrix is a 3-dimensional :class:`numpy.ndarray` which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0, so a11 can be accessed by 
        taking the slice a[:,0,0].  
        

        Returns
        ---------
        a : complex :class:`numpy.ndarray` of shape `fxnxn`
                the active scattering parameter matrix.
        
        See Also
        ------------
        s 
        y
        z
        t
        a
        '''
        return 1/self.s
        
    @a.setter
    def a(self, value):
        raise (NotImplementedError)
    
    
        
    @property
    def z0(self):
        '''
        Characteristic impedance[s] of the network ports.

        This property stores the  characteristic impedance of each port
        of the network. Because it is possible that each port has
        a different characteristic impedance each varying with
        frequency, `z0` is stored internally as a `fxn` array.

        However because  `z0` is frequently simple (like 50ohm), it can
        be set with just number as well.

        Returns
        --------
        z0 : :class:`numpy.ndarray` of shape fxn
                characteristic impedance for network
        
        '''
        # i hate this function
        # it was written this way because id like to allow the user to
        # set the z0 before the s-parameters are set. However, in this
        # case we dont know how to re-shape the z0 to fxn. to solve this
        # i attempt to do the re-shaping when z0 is accessed, not when
        # it is set. this is what makes this function confusing.
        try:
            if len(npy.shape(self._z0)) ==0:
                try:
                    #try and re-shape z0 to match s
                    self._z0=self._z0*npy.ones(self.s.shape[:-1])
                except(AttributeError):
                    print ('Warning: Network has improper \'z0\' shape.')
                    #they have yet to set s .

            elif len(npy.shape(self._z0)) ==1:
                try:
                    if len(self._z0) == self.frequency.npoints:
                        # this z0 is frequency dependent but not port dependent
                        self._z0 = \
                                npy.repeat(npy.reshape(self._z0,(-1,1)),self.number_of_ports,1)

                    elif len(self._z0) == self.number_of_ports:
                        # this z0 is port dependent but not frequency dependent
                        self._z0 = self._z0*npy.ones(\
                                (self.frequency.npoints,self.number_of_ports))

                    else:
                        raise(IndexError('z0 has bad shape'))

                except(AttributeError):
                    # there is no self.frequency, or self.number_of_ports
                    raise(AttributeError('Error: i cant reshape z0 through inspection. you must provide correctly shaped z0, or s-matrix first.'))

            return self._z0

        except(AttributeError):
            print 'Warning: z0 is undefined. Defaulting to 50.'
            self.z0=50
            return self.z0 #this is not an error, its a recursive call

    @z0.setter
    def z0(self, z0):
        '''z0=npy.array(z0)
        if len(z0.shape) < 2:
                try:
                        #try and re-shape z0 to match s
                        z0=z0*npy.ones(self.s.shape[:-1])
                except(AttributeError):
                        print ('Warning: you should store a Network\'s \'s\' matrix before its \'z0\'')
                        #they have yet to set s .
                        pass
        '''
        self._z0 = npy.array(z0,dtype=complex)

    @property
    def frequency(self):
        '''
        frequency information for the network.

        This property is a :class:`~skrf.frequency.Frequency` object.
        It holds the frequency vector, as well frequency unit, and
        provides other properties related to frequency information, such
        as start, stop, etc.

        Returns
        --------
        frequency :  :class:`~skrf.frequency.Frequency` object
                frequency information for the network.


        See Also
        ---------
        f : property holding frequency vector in Hz
        change_frequency : updates frequency property, and
            interpolates s-parameters if needed
        interpolate : interpolate function based on new frequency
            info
        '''
        try:
            return self._frequency
        except (AttributeError):
            self._frequency = Frequency(0,0,0)
            return self._frequency

    @frequency.setter
    def frequency(self, new_frequency):
        '''
        takes a Frequency object, see  frequency.py
        '''
        if isinstance(new_frequency, Frequency):
            self._frequency = new_frequency.copy()
        else:
            try:
                self._frequency = Frequency.from_f(new_frequency)
            except (TypeError):
                raise TypeError('Could not convert argument to a frequency vector')

    

    @property
    def inv(self):
        '''
        a :class:`Network` object with 'inverse' s-parameters.

        This is used for de-embeding. It is defined so that the inverse
        of a Network cascaded with itself is unity.

        Returns
        ---------
        inv : a :class:`Network` object
                a :class:`Network` object with 'inverse' s-parameters.

        See Also
        ----------
                inv : function which implements the inverse s-matrix
        '''
        if self.number_of_ports <2:
            raise(TypeError('One-Port Networks dont have inverses'))
        out = self.copy()
        out.s = inv(self.s)
        return out

    @property
    def f(self):
        '''
        the frequency vector for the network, in Hz.

        Returns
        --------
        f : :class:`numpy.ndarray`
                frequency vector in Hz

        See Also
        ---------
                frequency : frequency property that holds all frequency
                        information
        '''
        return self.frequency.f

    @f.setter
    def f(self,f):
        tmpUnit= self.frequency.unit
        self.frequency = Frequency.from_f(f, unit=tmpUnit)

    
    ## SECONDARY PROPERTIES
    @property
    def number_of_ports(self):
        '''
        the number of ports the network has.

        Returns
        --------
        number_of_ports : number
                the number of ports the network has.

        '''
        try:
            return self.s.shape[1]
        except (AttributeError):
            return 0

    @property
    def nports(self):
        '''
        the number of ports the network has.

        Returns
        --------
        number_of_ports : number
                the number of ports the network has.

        '''
        return self.number_of_ports
        
    @property
    def passivity(self):
        '''
        passivity metric for a multi-port network.

        This returns a matrix who's diagonals are equal to the total
        power received at all ports, normalized to the power at a single
        excitement port.

        mathmatically, this is a test for unitary-ness of the
        s-parameter matrix [#]_.

        for two port this is

        .. math::

                ( |S_{11}|^2 + |S_{21}|^2 \, , \, |S_{22}|^2+|S_{12}|^2)

        in general it is

        .. math::

                S^H \\cdot S

        where :math:`H` is conjugate transpose of S, and :math:`\\cdot`
        is dot product.

        Returns
        ---------
        passivity : :class:`numpy.ndarray` of shape fxnxn

        References
        ------------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Lossless_networks
        '''
        if self.number_of_ports == 1:
            raise (ValueError('Doesnt exist for one ports'))

        pas_mat = self.s.copy()
        for f in range(len(self.s)):
            pas_mat[f,:,:] = npy.dot(self.s[f,:,:].conj().T, self.s[f,:,:])
		
        return pas_mat
    
    
    
    
	## NETWORK CLASIFIER
	def is_reciprocal(self):
		'''
		test for reciprocity
		'''
		raise(NotImplementedError)
	
	def is_symmetric(self):
		'''
		test for symmetry
		'''
		raise(NotImplementedError)
	
	def is_passive(self):
		'''
		test for passivity 
		'''
		raise(NotImplementedError)		
	
	def is_lossless(self):
		'''
		test for losslessness
		'''
		raise(NotImplementedError)	
    
    ## CLASS METHODS
    def copy(self):
        '''
        Returns a copy of this Network
        
        Needed to allow pass-by-value for a Network instead of 
        pass-by-reference
        '''
        ntwk = Network()
        ntwk.frequency = self.frequency.copy()
        ntwk.s = self.s.copy()
        ntwk.z0 = self.z0.copy()
        ntwk.name = self.name
        return ntwk
    
    def copy_from(self,other):
        '''
        Copies the contents of another Network into self
        
        Uses copy, so that the data is passed-by-value, not reference
        
        Parameters
        -----------
        other : Network 
            the network to copy the contents of
        
        Examples
        -----------
        >>> a = rf.N()
        >>> b = rf.N('my_file.s2p')
        >>> a.copy_from (b)
        '''
        for attr in ['_s','frequency','_z0','name' ]:
            self.__setattr__(attr,copy(other.__getattribute__(attr)))
    
    
    # touchstone file IO
    def read_touchstone(self, filename):
        '''
        loads values from a touchstone file.

        The work of this function is done through the
        :class:`~skrf.io.touchstone` class.

        Parameters
        ----------
        filename : str or file-object
            touchstone file name.


        Notes
        ------
        only the scattering parameters format is supported at the
        moment



        '''
        from io import touchstone
        touchstoneFile = touchstone.Touchstone(filename)
        
        if touchstoneFile.get_format().split()[1] != 's':
            raise NotImplementedError('only s-parameters supported for now.')

        self.comments = touchstoneFile.get_comments()        

        # set z0 before s so that y and z can be computed
        self.z0 = complex(touchstoneFile.resistance)  
        f, self.s = touchstoneFile.get_sparameter_arrays() # note: freq in Hz
        self.frequency = Frequency.from_f(f, unit='hz')
        self.frequency.unit = touchstoneFile.frequency_unit 
        
        try:
            self.name = os.path.basename( os.path.splitext(filename)[0])
            # this may not work if filename is a file object
        except(AttributeError):
            # in case they pass a file-object instead of file name, 
            # get the name from the touchstone file
            try: 
                self.name = os.path.basename( os.path.splitext(touchstoneFile.filename)[0])
            except():
                print 'warning: couldnt inspect network name'
                self.name=''
            pass
        #TODO: add Network property `comments` which is read from
        # touchstone file. 
    
    def write_touchstone(self, filename=None, dir = './', write_z0=False):
        '''
        write a contents of the :class:`Network` to a touchstone file.


        Parameters
        ----------
        filename : a string, optional
                touchstone filename, without extension. if 'None', then
                will use the network's :attr:`name`.
        dir : string, optional
                the directory to save the file in. Defaults
                to cwd './'.
        write_z0 : boolean
            write impedance information into touchstone as comments, 
            like Ansoft HFSS does

        Notes
        -------
                format supported at the moment is,
                        HZ S RI

                The functionality of this function should take place in the
                :class:`~skrf.touchstone.touchstone` class.


        '''
        if filename is None:
            if self.name is not None:
                filename= self.name
            else:
                raise ValueError('No filename given. Network must have a name, or you must provide a filename')

        extension = '.s%ip'%self.number_of_ports

        outputFile = open(dir+'/'+filename+extension,"w")
        
        # Add '!' Touchstone comment delimiters to the start of every line
        # in self.comments
        commented_header = ''
        if self.comments:
            for comment_line in self.comments.split('\n'):
                commented_header += '!{}\n'.format(comment_line)

        # write header file.
        # the '#'  line is NOT a comment it is essential and it must be
        #exactly this format, to work
        # [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
        outputFile.write('!Created with skrf (http://scikit-rf.org).\n')
        outputFile.write(commented_header)
        outputFile.write('# ' + self.frequency.unit + ' S RI R ' + str(self.z0[0,0]) +" \n")

        #write comment line for users (optional)
        outputFile.write ("!freq\t")
        for n in range(self.number_of_ports):
            for m in range(self.number_of_ports):
                outputFile.write("Re" +'S'+`m+1`+ `n+1`+  "\tIm"+\
                'S'+`m+1`+ `n+1`+'\t')
        outputFile.write('\n')

        # write out data, note: this could be done with matrix
        #manipulations, but its more readable to me this way
        for f in range(len(self.f)):
            outputFile.write(str(self.frequency.f_scaled[f])+'\t')

            for n in range(self.number_of_ports):
                for m in range(self.number_of_ports):
                    outputFile.write( str(npy.real(self.s[f,m,n])) + '\t'\
                     + str(npy.imag(self.s[f,m,n])) +'\t')

            # write out the z0 following hfss's convention if desired
            if write_z0:
                outputFile.write('\n')
                outputFile.write('! Port Impedance\t' )
                for n in range(self.number_of_ports):
                    outputFile.write('%.14f\t%.14f\t'%(self.z0[f,n].real, self.z0[f,n].imag))
            outputFile.write('\n')

        outputFile.close()

    def write(self, file=None, *args, **kwargs):
        '''
        Write the Network to disk using the :mod:`pickle` module.
        
        The resultant file can be read either by using the Networks 
        constructor, :func:`__init__` , the read method :func:`read`, or 
        the general read function :func:`skrf.io.general.read`
        
        
        Parameters
        -----------
        file : str or file-object
            filename or a file-object. If left as None then the 
            filename will be set to Network.name, if its not None. 
            If both are None, ValueError is raised.
        \*args, \*\*kwargs : 
            passed through to :func:`~skrf.io.general.write`
        
        Notes
        ------
        If the self.name is not None and file is  can left as None
        and the resultant file will have the `.ntwk` extension appended
        to the filename. 
        
        Examples
        ---------
        >>> n = rf.N(f=[1,2,3],s=[1,1,1],z0=50, name = 'open')
        >>> n.write()
        >>> n2 = rf.read('open.ntwk')
        
        See Also
        ---------
        skrf.io.general.write : write any skrf object
        skrf.io.general.read : read any skrf object
        '''
        # this import is delayed untill here because of a circular depency
        from io.general import write
        
        if file is None:
            if self.name is None:
                 raise (ValueError('No filename given. You must provide a filename, or set the name attribute'))
            file = self.name

        write(file,self,*args, **kwargs)
    
    
    def read(self, *args, **kwargs):
        '''
        Read a Network from a 'ntwk' file
        
        A ntwk file is written with :func:`write`. It is just a pickled
        file. 
        
        Parameters
        -------------
        \*args, \*\*kwargs : args and kwargs 
            passed to :func:`skrf.io.general.write`
        
        Notes
        ------
        This function calls :func:`skrf.io.general.read`.
        
        Examples
        -----------
        >>> rf.read('myfile.ntwk')
        >>> rf.read('myfile.p')
            
        See Also
        ----------
        write
        skrf.io.general.write
        skrf.io.general.read
        '''
        from io.general import read
        self.copy_from(read(*args, **kwargs))
        
    
    # interpolation
    def interpolate(self, new_frequency,**kwargs):
        '''
        Return an interpolated network, from a new :class:'~skrf.frequency.Frequency'.

        Interpolate the networks s-parameters linearly in real and 
        imaginary components. Other interpolation types can be used 
        by passing appropriate `\*\*kwargs`. This function `returns` an 
        interpolated Network. Alternatively :func:`~Network.interpolate_self` 
        will interpolate self.
        

        Parameters
        -----------
        new_frequency : :class:`~skrf.frequency.Frequency`
            frequency information to interpolate 
        **kwargs : keyword arguments
            passed to :func:`scipy.interpolate.interp1d` initializer.

        Returns
        ----------
        result : :class:`Network`
                an interpolated Network

        Notes
        --------
        See  :func:`scipy.interpolate.interpolate.interp1d` for useful 
        kwargs. For example
            **kind** : str or int
                Specifies the kind of interpolation as a string ('linear',
                'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or
                as an integer specifying the order of the spline 
                interpolator to use.
        
        See Also
        ----------
        resample
        interpolate_self 
        interpolate_from_f
        
        Examples
        -----------
        .. ipython::
        
            @suppress
            In [21]: import skrf as rf 
            
            In [21]: n = rf.data.ring_slot 
            
            In [21]: n
            
            In [21]: new_freq = rf.Frequency(75,110,501,'ghz')
            
            In [21]: n.interpolate(new_freq, kind = 'cubic')
        
        '''
        # create interpolation objects
        interpolation_s_re = \
            interp1d(self.frequency.f,self.s_re,axis=0,**kwargs)
        interpolation_s_im = \
            interp1d(self.frequency.f,self.s_im,axis=0,**kwargs)
        interpolation_z0_re = \
            interp1d(self.frequency.f,self.z0.real,axis=0,**kwargs)
        interpolation_z0_im = \
            interp1d(self.frequency.f,self.z0.imag,axis=0,**kwargs)

        # make new network and fill with interpolated s, and z0
        result = self.copy()
        result.frequency = new_frequency
        result.s = interpolation_s_re(new_frequency.f) +\
            1j*interpolation_s_im(new_frequency.f)
        result.z0 = interpolation_z0_re(new_frequency.f) +\
            1j*interpolation_z0_im(new_frequency.f)

        return result

    def interpolate_self_npoints(self, npoints, **kwargs):
        '''
        Interpolate network based on a new number of frequency points

        
        Parameters
        ----------
        npoints : int
                number of frequency points
        **kwargs : keyword arguments
                passed to :func:`scipy.interpolate.interp1d` initializer.

        See Also
        ---------
        interpolate_self : same functionality but takes a Frequency
                object
        interpolate : same functionality but takes a Frequency
                object and returns a new Network, instead of updating
                itself.
        
        Notes
        -------
        The function :func:`~Network.resample` is an alias for 
        :func:`~Network.interpolate_self_npoints`. 
        
        Examples
        -----------
        .. ipython::
        
            @suppress
            In [21]: import skrf as rf 
            
            In [21]: n = rf.data.ring_slot 
            
            In [21]: n
            
            In [21]: n.resample(501) # resample is an alias
            
            In [21]: n
            
        '''
        new_frequency = self.frequency.copy()
        new_frequency.npoints = npoints
        self.interpolate_self(new_frequency, **kwargs)

    ##convenience
    resample = interpolate_self_npoints
    
    def interpolate_self(self, new_frequency, **kwargs):
        '''
        Interpolates s-parameters given a new
        :class:'~skrf.frequency.Frequency' object.

        See :func:`~Network.interpolate` for more information. 

        Parameters
        -----------
        new_frequency : :class:`~skrf.frequency.Frequency`
                frequency information to interpolate at
        **kwargs : keyword arguments
                passed to :func:`scipy.interpolate.interp1d` initializer.

        See Also
        ----------
        resample
        interpolate
        interpolate_from_f
        '''
        ntwk = self.interpolate(new_frequency, **kwargs)
        self.frequency, self.s,self.z0 = ntwk.frequency, ntwk.s,ntwk.z0
    
    def interpolate_from_f(self, f, interp_kwargs={}, **kwargs):
        '''
        Interpolates s-parameters from a frequency vector.
        
        Given a frequency vector, and optionally a `unit` (see \*\*kwargs)
        , interpolate the networks s-parameters linearly in real and 
        imaginary components. 
        
        See :func:`~Network.interpolate` for more information. 

    
        

        Parameters
        -----------
        new_frequency : :class:`~skrf.frequency.Frequency`
            frequency information to interpolate at
        interp_kwargs : 
            dictionary of kwargs to be passed through to 
            :func:`scipy.interpolate.interpolate.interp1d`
        \*\*kwargs : 
            passed to :func:`scipy.interpolate.interp1d` initializer.
            
        Notes
        ---------
        This creates a new :class:`~skrf.frequency.Frequency`, object 
        using the method :func:`~skrf.frequency.Frequency.from_f`, and then calls
        :func:`~Network.interpolate_self`.
        
        See Also
        ----------
        resample
        interpolate
        interpolate_self 


        '''
        freq = Frequency.from_f(f,**kwargs)
        self.interpolate_self(freq, **interp_kwargs)
        
        
    def flip(self):
        '''
        swaps the ports of a two port Network
        '''
        if self.number_of_ports == 2:
            self.renumber( [0,1], [1,0] )
        else:
            raise ValueError('you can only flip two-port Networks')

    def renumber(self, from_ports, to_ports):
        '''
        renumbers some ports of a two port Network

        Parameters
        -----------
        from_ports : list-like
        to_ports: list-like

        Examples
        ---------
        To flip the ports of a 2-port network 'foo':
        >>> foo.renumber( [0,1], [1,0] )

        To rotate the ports of a 3-port network 'bar' so that port 0 becomes port 1:
        >>> bar.renumber( [0,1,2], [1,2,0] )

        To swap the first and last ports of a network 'duck':
        >>> duck.renumber( [0,-1], [-1,0] )
        '''
        from_ports = npy.array(from_ports)
        to_ports = npy.array(to_ports)
        if len(npy.unique(from_ports)) != len(from_ports):
            raise ValueError('an index can appear at most once in from_ports or to_ports')
        if any(npy.unique(from_ports) != npy.unique(to_ports)):
            raise ValueError('from_ports and to_ports must have the same set of indices')

        self.s[:,to_ports,:] = self.s[:,from_ports,:]  # renumber rows
        self.s[:,:,to_ports] = self.s[:,:,from_ports]  # renumber columns
        self.z0[:,to_ports] = self.z0[:,from_ports]

    # plotting
    def plot_s_smith(self,m=None, n=None,r=1,ax = None, show_legend=True,\
            chart_type='z', draw_labels=False, label_axes=False, *args,**kwargs):
        '''
        plots the scattering parameter on a smith chart

        plots indices `m`, `n`, where `m` and `n` can be integers or
        lists of integers.


        Parameters
        -----------
        m : int, optional
                first index
        n : int, optional
                second index
        ax : matplotlib.Axes object, optional
                axes to plot on. in case you want to update an existing
                plot.
        show_legend : boolean, optional
                to turn legend show legend of not, optional
        chart_type : ['z','y']
            draw impedance or addmitance contours 
        draw_labels : Boolean 
            annotate chart with impedance values 
        label_axes : Boolean
            Label axis with titles `Real` and `Imaginary`
        border : Boolean 
            draw rectangular border around image with ticks
        
        \*args : arguments, optional
                passed to the matplotlib.plot command
        \*\*kwargs : keyword arguments, optional
                passed to the matplotlib.plot command


        See Also
        --------
        plot_vs_frequency_generic - generic plotting function
        smith -  draws a smith chart

        Examples
        ---------
        >>> myntwk.plot_s_smith()
        >>> myntwk.plot_s_smith(m=0,n=1,color='b', marker='x')
        '''
        # TODO: prevent this from re-drawing smith chart if one alread
        # exists on current set of axes

        # get current axis if user doesnt supply and axis
        if ax is None:
            ax = plb.gca()


        if m is None:
            M = range(self.number_of_ports)
        else:
            M = [m]
        if n is None:
            N = range(self.number_of_ports)
        else:
            N = [n]

        if 'label'  not in kwargs.keys():
            generate_label=True
        else:
            generate_label=False

        for m in M:
            for n in N:
                # set the legend label for this trace to the networks name if it
                # exists, and they didnt pass a name key in the kwargs
                if generate_label:
                    if self.name is None:
                        if plb.rcParams['text.usetex']:
                            label_string = '$S_{'+repr(m+1) + repr(n+1)+'}$'
                        else:
                            label_string = 'S'+repr(m+1) + repr(n+1)
                    else:
                        if plb.rcParams['text.usetex']:
                            label_string = self.name+', $S_{'+repr(m+1) + \
                                    repr(n+1)+'}$'
                        else:
                            label_string = self.name+', S'+repr(m+1) + repr(n+1)

                    kwargs['label'] = label_string

                # plot the desired attribute vs frequency
                if len (ax.patches) == 0:
                    smith(ax=ax, smithR = r, chart_type=chart_type, draw_labels=draw_labels)
                ax.plot(self.s[:,m,n].real,  self.s[:,m,n].imag, *args,**kwargs)

        #draw legend
        if show_legend:
            ax.legend()
        ax.axis(npy.array([-1.1,1.1,-1.1,1.1])*r)
        
        if label_axes:
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')

    def plot_passivity(self,port=None, ax = None, show_legend=True,*args,**kwargs):
        '''
        plots the passivity of a network, possibly for a specific port.


        Parameters
        -----------
        port: int
                calculate passivity of a given port
        ax : matplotlib.Axes object, optional
                axes to plot on. in case you want to update an existing
                plot.
        show_legend : boolean, optional
                to turn legend show legend of not, optional
        *args : arguments, optional
                passed to the matplotlib.plot command
        **kwargs : keyword arguments, optional
                passed to the matplotlib.plot command


        See Also
        --------
        plot_vs_frequency_generic - generic plotting function
        passivity - passivity property

        Examples
        ---------
        >>> myntwk.plot_s_rad()
        >>> myntwk.plot_s_rad(m=0,n=1,color='b', marker='x')
        '''
        if port is None:
            port = range(self.number_of_ports)

        for mn in list(port):
            self.plot_vs_frequency_generic(attribute= 'passivity',\
                    y_label='Passivity', m=mn,n=mn, ax=ax,\
                    show_legend = show_legend,*args,**kwargs)

    def plot_it_all(self,*args, **kwargs):
        plb.subplot(221)
        getattr(self,'plot_s_db')(*args, **kwargs)
        plb.subplot(222)
        getattr(self,'plot_s_deg')(*args, **kwargs)
        plb.subplot(223)
        getattr(self,'plot_s_smith')(*args, **kwargs)
        plb.subplot(224)
        getattr(self,'plot_s_complex')(*args, **kwargs)

    # noise
    def add_noise_polar(self,mag_dev, phase_dev,**kwargs):
        '''
        adds a complex zero-mean gaussian white-noise.

        adds a complex zero-mean gaussian white-noise of a given
        standard deviation for magnitude and phase

        Parameters
        ------------
        mag_dev : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase [in degrees]

        '''
        phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = self.s.shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = self.s.shape)
        phase = (self.s_deg+phase_rv)
        mag = self.s_mag + mag_rv
        self.s = mag* npy.exp(1j*npy.pi/180.*phase)

    def add_noise_polar_flatband(self,mag_dev, phase_dev,**kwargs):
        '''
        adds a flatband complex zero-mean gaussian white-noise signal of
        given standard deviations for magnitude and phase

        Parameters
        ------------
        mag_dev : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase [in degrees]

        '''
        phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = self.s[0].shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = self.s[0].shape)

        phase = (self.s_deg+phase_rv)
        mag = self.s_mag + mag_rv
        self.s = mag* npy.exp(1j*npy.pi/180.*phase)

    def multiply_noise(self,mag_dev, phase_dev, **kwargs):
        '''
        multiplys a complex bivariate gaussian white-noise signal
        of given standard deviations for magnitude and phase.
        magnitude mean is 1, phase mean is 0

        takes:
                mag_dev: standard deviation of magnitude
                phase_dev: standard deviation of phase [in degrees]
                n_ports: number of ports. defualt to 1
        returns:
                nothing
        '''
        phase_rv = stats.norm(loc=0, scale=phase_dev).rvs(\
                size = self.s.shape)
        mag_rv = stats.norm(loc=1, scale=mag_dev).rvs(\
                size = self.s.shape)
        self.s = mag_rv*npy.exp(1j*npy.pi/180.*phase_rv)*self.s

    def nudge(self, amount=1e-12):
        '''
        Perturb s-parameters by small amount. 
        
        This is useful to work-around numerical bugs.
        
        Notes
        -----------
        This function is  
            self.s = self.s + 1e-12

        Parameters
        ------------
        amount : number,
                amount to add to s parameters

        '''
        self.s = self.s + amount

## Functions operating on Network[s]
def connect(ntwkA, k, ntwkB, l, num=1):
    '''
    connect two n-port networks together.

    specifically, connect ports `k` thru `k+num-1` on `ntwkA` to ports
    `l` thru `l+num-1` on `ntwkB`. The resultant network has
    (ntwkA.nports+ntwkB.nports-2*num) ports. The port indices ('k','l')
    start from 0. Port impedances **are** taken into account.

    Parameters
    -----------
    ntwkA : :class:`Network`
            network 'A'
    k : int
            starting port index on `ntwkA` ( port indices start from 0 )
    ntwkB : :class:`Network`
            network 'B'
    l : int
            starting port index on `ntwkB`
    num : int
            number of consecutive ports to connect (default 1)


    Returns
    ---------
    ntwkC : :class:`Network`
            new network of rank (ntwkA.nports + ntwkB.nports - 2*num)


    See Also
    -----------
            connect_s : actual  S-parameter connection algorithm.
            innerconnect_s : actual S-parameter connection algorithm.

    Notes
    -------
            the effect of mis-matched port impedances is handled by inserting
            a 2-port 'mismatch' network between the two connected ports.
            This mismatch Network is calculated with the
            :func:`impedance_mismatch` function.

    Examples
    ---------
    To implement a *cascade* of two networks

    >>> ntwkA = rf.Network('ntwkA.s2p')
    >>> ntwkB = rf.Network('ntwkB.s2p')
    >>> ntwkC = rf.connect(ntwkA, 1, ntwkB,0)

    '''
    # some checking 
    check_frequency_equal(ntwkA,ntwkB)
    
    # create output Network, from copy of input 
    ntwkC = ntwkA.copy()
    
    # if networks' z0's are not identical, then connect a impedance
    # mismatch, which takes into account the effect of differing port
    # impedances. 
    #import pdb;pdb.set_trace()
    if assert_z0_at_ports_equal(ntwkA,k,ntwkB,l) == False:
        ntwkC.s = connect_s(
            ntwkA.s, k, 
            impedance_mismatch(ntwkA.z0[:,k], ntwkB.z0[:,l]), 0)
        # the connect_s() put the mismatch's output port at the end of
        #   ntwkC's ports.  Fix the new port's impedance, then insert it
        #   at position k where it belongs.
        ntwkC.z0[:,k:] = npy.hstack((ntwkC.z0[:,k+1:], ntwkB.z0[:,[l]]))
        ntwkC.renumber(from_ports= [ntwkC.nports-1] + range(k, ntwkC.nports-1),
                       to_ports=range(k, ntwkC.nports))

    # call s-matrix connection function
    ntwkC.s = connect_s(ntwkC.s,k,ntwkB.s,l)

    # combine z0 arrays and remove ports which were `connected`
    ntwkC.z0 = npy.hstack(
        (npy.delete(ntwkA.z0, range(k,k+num), 1), npy.delete(ntwkB.z0, range(l,l+num), 1)))

    # if we're connecting more than one port, call innerconnect to finish the job
    if num>1:
        ntwkC = innerconnect(ntwkC, k, ntwkA.nports-1+l, num-1)

    return ntwkC

def connect2(ntwkA, k, ntwkB, l, num=1):
    '''
    connect two n-port networks together (alternative implementation)

    specifically, connect ports `k` thru `k+num-1` on `ntwkA` to ports
    `l` thru `l+num-1` on `ntwkB`. The resultant network has
    (ntwkA.nports+ntwkB.nports-2*num) ports. The port indices ('k','l')
    start from 0. Port impedances **are** taken into account.

    Parameters
    -----------
    ntwkA : :class:`Network`
            network 'A'
    k : int
            starting port index on `ntwkA` ( port indices start from 0 )
    ntwkB : :class:`Network`
            network 'B'
    l : int
            starting port index on `ntwkB`
    num : int
            number of consecutive ports to connect (default 1)


    Returns
    ---------
    ntwkC : :class:`Network`
            new network of rank (ntwkA.nports + ntwkB.nports - 2*num)


    See Also
    -----------
            connect_s : actual  S-parameter connection algorithm.
            innerconnect_s : actual S-parameter connection algorithm.

    Notes
    -------
            the effect of mis-matched port impedances is handled by inserting
            a 2-port 'mismatch' network between the two connected ports.
            This mismatch Network is calculated with the
            :func:impedance_mismatch function.

    Examples
    ---------
    To implement a *cascade* of two networks

    >>> ntwkA = rf.Network('ntwkA.s2p')
    >>> ntwkB = rf.Network('ntwkB.s2p')
    >>> ntwkC = rf.connect(ntwkA, 1, ntwkB,0)

    '''
    # some checking 
    check_frequency_equal(ntwkA,ntwkB)
    
    # create output Network, from copy of input 
    ntwkC = ntwkA.copy()
    
    # call s-matrix connection function
    ntwkC.s = connect_s(ntwkC.s,k,ntwkB.s,l)

    # combine z0 arrays and remove ports which were `connected`
    ntwkC.z0 = npy.hstack(
        (npy.delete(ntwkA.z0, range(k,k+num), 1), npy.delete(ntwkB.z0, range(l,l+num), 1)))

    # if we're connecting more than one port, call innerconnect to finish the job
    if num>1:
        ntwkC = innerconnect(ntwkC, k, ntwkA.nports-1+l, num-1)

    return ntwkC

def innerconnect(ntwkA, k, l, num=1):
    '''
    connect ports of a single n-port network.

    this results in a (n-2)-port network. remember port indices start
    from 0.

    Parameters
    -----------
    ntwkA : :class:`Network`
        network 'A'
    k,l : int
        starting port indices on ntwkA ( port indices start from 0 )
    num : int
        number of consecutive ports to connect

    Returns
    ---------
    ntwkC : :class:`Network`
        new network of rank (ntwkA.nports - 2*num)

    See Also
    -----------
        connect_s : actual  S-parameter connection algorithm.
        innerconnect_s : actual S-parameter connection algorithm.

    Notes
    -------
        a 2-port 'mismatch' network is inserted between the connected ports
        if their impedances are not equal.

    Examples
    ---------
    To connect ports '0' and port '1' on ntwkA

    >>> ntwkA = rf.Network('ntwkA.s3p')
    >>> ntwkC = rf.innerconnect(ntwkA, 0,1)

    '''
    # create output Network, from copy of input 
    ntwkC = ntwkA.copy()

    # connect a impedance mismatch, which will takes into account the
    # effect of differing port impedances
    if not (ntwkA.z0[:,k] == ntwkA.z0[:,l]).all():
        ntwkC.s = connect_s(\
            ntwkA.s,k, \
            impedance_mismatch(ntwkA.z0[:,k], ntwkA.z0[:,l]), 0)
        # the connect_s() put the mismatch's output port at the end of
        #   ntwkC's ports.  Fix the new port's impedance, then insert it
        #   at position k where it belongs.
        ntwkC.z0[:,k:] = npy.hstack((ntwkC.z0[:,k+1:], ntwkC.z0[:,[l]]))
        ntwkC.renumber(from_ports= [ntwkC.nports-1] + range(k, ntwkC.nports-1),
                       to_ports=range(k, ntwkC.nports))

    # call s-matrix connection function
    ntwkC.s = innerconnect_s(ntwkC.s,k,l)

    # update the characteristic impedance matrix
    ntwkC.z0 = npy.delete(ntwkC.z0, range(k,k+num) + range(l,l+num),1)

    # recur if we're connecting more than one port
    if num>1:
        ntwkC = innerconnect(ntwkC, k, l-1, num-1)

    return ntwkC

def cascade(ntwkA,ntwkB):
    '''
    Cascade two 2-port Networks together

    Connects port 1 of `ntwkA` to port 0 of `ntwkB`. This calls
    `connect(ntwkA,1, ntwkB,0)`, which is a more general function.

    Parameters
    -----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : Network
            network `ntwkB`

    Returns
    --------
    C : Network
            the resultant network of ntwkA cascaded with ntwkB

    See Also
    ---------
    connect : connects two Networks together at arbitrary ports.
    '''
    return connect(ntwkA,1, ntwkB,0)

def de_embed(ntwkA,ntwkB):
    '''
    De-embed `ntwkA` from `ntwkB`. 
    
    This calls `ntwkA.inv ** ntwkB`. The syntax of cascading an inverse
    is more explicit, it is recomended that it be used instead of this 
    function.

    Parameters
    -----------
    ntwkA : :class:`Network`
            network `ntwkA`
    ntwkB : :class:`Network`
            network `ntwkB`

    Returns
    --------
    C : Network
            the resultant network of  ntwkB de-embeded from ntwkA

    See Also
    ---------
    connect : connects two Networks together at arbitrary ports.

    '''
    return ntwkA.inv ** ntwkB

def stitch(ntwkA, ntwkB, **kwargs):
    '''
    Stitches ntwkA and ntwkB together.
    
    Concatenates  two networks' data.  Given two networks that cover 
    different frequency bands this can be used to combine their data 
    into a single network. 
    
    Parameters
    ------------
    ntwkA, ntwkB : :class:`Network` objects
        Networks to stitch together
    
    \*\*kwargs : keyword args
        passed to :class:`Network` constructor, for output network
    
    Returns
    ---------
    ntwkC : :class:`Network`
        result of stitching the networks `ntwkA` and `ntwkB` together
    
    Examples
    ----------
    >>> from skrf.data import wr2p2_line, wr1p5_line
    >>> rf.stitch(wr2p2_line, wr1p5_line)
    2-Port Network: 'wr2p2,line',  330-750 GHz, 402 pts, z0=[ 50.+0.j  50.+0.j]
    '''
    A,B = ntwkA, ntwkB
    C = Network(
        frequency = Frequency.from_f(npy.r_[A.f[:],B.f[:]], unit='hz'), 
        s = npy.r_[A.s,B.s],
        z0 = npy.r_[A.z0, B.z0],
        name = A.name,
        **kwargs
        )
    C.frequency.unit = A.frequency.unit
    return C

def average(list_of_networks):
    '''
    Calculates the average network from a list of Networks.

    This is complex average of the s-parameters for a  list of Networks. 


    Parameters
    -----------
    list_of_networks : list of :class:`Network` objects
        the list of networks to average

    Returns
    ---------
    ntwk : :class:`Network`
            the resultant averaged Network

    Notes
    ------
    This same function can be accomplished with properties of a
    :class:`~skrf.networkset.NetworkSet` class.

    Examples
    ---------

    >>> ntwk_list = [rf.Network('myntwk.s1p'), rf.Network('myntwk2.s1p')]
    >>> mean_ntwk = rf.average(ntwk_list)
    '''
    out_ntwk = list_of_networks[0].copy()

    for a_ntwk in list_of_networks[1:]:
        out_ntwk += a_ntwk

    out_ntwk.s = out_ntwk.s/(len(list_of_networks))

    return out_ntwk

def one_port_2_two_port(ntwk):
    '''
    calculates the two-port network given a  symetric, reciprocal and
    lossless one-port network.

    takes:
            ntwk: a symetric, reciprocal and lossless one-port network.
    returns:
            ntwk: the resultant two-port Network
    '''
    result = ntwk.copy()
    result.s = npy.zeros((result.frequency.npoints,2,2), dtype=complex)
    s11 = ntwk.s[:,0,0]
    result.s[:,0,0] = s11
    result.s[:,1,1] = s11
    ## HACK: TODO: verify this mathematically
    result.s[:,0,1] = npy.sqrt(1- npy.abs(s11)**2)*\
            npy.exp(1j*(npy.angle(s11)+npy.pi/2.*(npy.angle(s11)<0) -npy.pi/2*(npy.angle(s11)>0)))
    result.s[:,1,0] = result.s[:,0,1]
    return result

## Functions operating on s-parameter matrices
def connect_s(A,k,B,l):
    '''
    connect two n-port networks' s-matricies together.

    specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matricies. The function
    :func:`connect` operates on :class:`Network` types.

    Parameters
    -----------
    A : :class:`numpy.ndarray`
            S-parameter matrix of `A`, shape is fxnxn
    k : int
            port index on `A` (port indices start from 0)
    B : :class:`numpy.ndarray`
            S-parameter matrix of `B`, shape is fxnxn
    l : int
            port index on `B`

    Returns
    -------
    C : :class:`numpy.ndarray`
        new S-parameter matrix


    Notes
    -------
    internally, this function creates a larger composite network
    and calls the  :func:`innerconnect_s` function. see that function for more
    details about the implementation

    See Also
    --------
        connect : operates on :class:`Network` types
        innerconnect_s : function which implements the connection
            connection algorithm


    '''
 
    if k > A.shape[-1]-1 or l > B.shape[-1] - 1:
        raise(ValueError('port indices are out of range'))

    nf = A.shape[0]     # num frequency points
    nA = A.shape[1]     # num ports on A
    nB = B.shape[1]     # num ports on B
    nC = nA + nB        # num ports on C 
    
    #create composite matrix, appending each sub-matrix diagonally
    C = npy.zeros((nf, nC, nC), dtype='complex')
    C[:, :nA, :nA] = A.copy()
    C[:, nA:, nA:] = B.copy()

    # call innerconnect_s() on composit matrix C 
    return innerconnect_s(C, k, nA + l)

def innerconnect_s(A, k, l):
    '''
    connect two ports of a single n-port network's s-matrix.

    Specifically, connect port `k`  to port `l` on `A`. This results in
    a (n-2)-port network.  This     function operates on, and returns
    s-matricies. The function :func:`innerconnect` operates on
    :class:`Network` types.

    Parameters
    -----------
    A : :class:`numpy.ndarray`
        S-parameter matrix of `A`, shape is fxnxn
    k : int
        port index on `A` (port indices start from 0)
    l : int
        port index on `A`

    Returns
    -------
    C : :class:`numpy.ndarray`
            new S-parameter matrix

    Notes
    -----
    The algorithm used to calculate the resultant network is called a
    'sub-network growth',  can be found in [#]_. The original paper
    describing the  algorithm is given in [#]_.

    References
    ----------
    .. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

    .. [#] Filipsson, Gunnar; , "A New General Computer Algorithm for S-Matrix Calculation of Interconnected Multiports," Microwave Conference, 1981. 11th European , vol., no., pp.700-704, 7-11 Sept. 1981. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4131699&isnumber=4131585


    '''
    
    if k > A.shape[-1] - 1 or l > A.shape[-1] - 1:
        raise(ValueError('port indices are out of range'))

    nA = A.shape[1]  # num of ports on input s-matrix
    # create an empty s-matrix, to store the result
    C = npy.zeros(shape=A.shape, dtype='complex')

    # loop through ports and calulates resultant s-parameters
    for i in range(nA):
        for j in range(nA):
            C[:,i,j] = \
                A[:,i,j] + \
                ( A[:,k,j] * A[:,i,l] * (1 - A[:,l,k]) + \
                A[:,l,j] * A[:,i,k] * (1 - A[:,k,l]) +\
                A[:,k,j] * A[:,l,l] * A[:,i,k] + \
                A[:,l,j] * A[:,k,k] * A[:,i,l])/\
                ((1 - A[:,k,l]) * (1 - A[:,l,k]) - A[:,k,k] * A[:,l,l])

    # remove ports that were `connected`
    C = npy.delete(C, (k,l), 1)
    C = npy.delete(C, (k,l), 2)

    return C
   
## network parameter conversion       
def s2z(s,z0=50):
    '''
    Convert scattering parameters [#]_ to impedance parameters [#]_


    .. math::
        z = \\sqrt {z_0} \\cdot (I + s) (I - s)^{-1} \\cdot \\sqrt{z_0}

    Parameters
    ------------
    s : complex array-like
        scattering parameters
    z0 : complex array-like or number 
        port impedances                                         

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
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    
    '''
    if npy.isscalar(z0):
        z0 = npy.array(s.shape[0]*[s.shape[1] * [z0]])
    z = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    s = s.copy() # to prevent the original array from being altered
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

    if npy.isscalar(z0):
        z0 = npy.array(s.shape[0]*[s.shape[1] * [z0]])
    y = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    s = s.copy() # to prevent the original array from being altered
    s[s==-1.] = -1. + 1e-12 # solve numerical singularity
    s[s==1.] = 1. + 1e-12 # solve numerical singularity
    for fidx in xrange(s.shape[0]):
        sqrty0 = npy.mat(npy.sqrt(npy.diagflat(1.0/z0[fidx])))
        y[fidx] = sqrty0*(I-s[fidx])*(I+s[fidx])**-1*sqrty0
    return y

def s2t(s):
    '''
    Converts scattering parameters [#]_ to scattering transfer parameters [#]_ .

    transfer parameters are also refered to as
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
    #TODO: add docstring describing the mathematics of this
    #TODO: check rank(s) ==2
    # although unintuitive this is calculated by
    # [[s11, s21],[s12,s22]].T
    t = npy.array([
        [-1*(s[:,0,0]*s[:,1,1]- s[:,1,0]*s[:,0,1])/s[:,1,0],
            -s[:,1,1]/s[:,1,0]],
        [s[:,0,0]/s[:,1,0],
            1/s[:,1,0] ]
        ]).transpose()
    return t   

def z2s(z, z0=50):
    '''
    convert impedance parameters [#]_ to scattering parameters [#]_

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
    .. [#] http://en.wikipedia.org/wiki/S-parameters
    '''
    if npy.isscalar(z0):
        z0 = npy.array(z.shape[0]*[z.shape[1] * [z0]])
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
    if npy.isscalar(z0):
        z0 = npy.array(y.shape[0]*[y.shape[1] * [z0]])
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

## cascading assistance functions
def inv(s):
    '''
    Calculates 'inverse' s-parameter matrix, used for de-embeding

    This is not literally the inverse of the s-parameter matrix. Instead, it
    is defined such that the inverse of the s-matrix cascaded
    with itself is unity.

    .. math::

            inv(s) = t2s({s2t(s)}^{-1})

    where :math:`x^{-1}` is the matrix inverse. In words, this
    is the inverse of the scattering transfer parameters matrix
    transformed into a scattering parameters matrix.

    Parameters
    -----------
    s : :class:`numpy.ndarray` (shape fx2x2)
            scattering parameter matrix.

    Returns
    -------
    s' : :class:`numpy.ndarray`
            inverse scattering parameter matrix.

    See Also
    ---------
    t2s : converts scattering transfer parameters to scattering parameters
    s2t : converts scattering parameters to scattering transfer parameters


    '''
    # this idea is from lihan
    i = s2t(s) 
    for f in range(len(i)):
        i[f,:,:] = npy.linalg.inv(i[f,:,:])   # could also be written as
                                              #   npy.mat(i[f,:,:])**-1  -- Trey
    i = t2s(i)
    return i

def flip(a):
    '''
    invert the ports of a networks s-matrix, 'flipping' it over

    Parameters
    -----------
    a : :class:`numpy.ndarray`
            scattering parameter matrix. shape should be should be 2x2, or
            fx2x2

    Returns
    -------
    a' : :class:`numpy.ndarray`
            flipped scattering parameter matrix, ie interchange of port 0
            and port 1

    Note
    -----
                    only works for 2-ports at the moment
    '''
    c = a.copy()

    if len (a.shape) > 2 :
        for f in range(a.shape[0]):
            c[f,:,:] = flip(a[f,:,:])
    elif a.shape == (2,2):
        c[0,0] = a[1,1]
        c[1,1] = a[0,0]
        c[0,1] = a[1,0]
        c[1,0] = a[0,1]
    else:
        raise IndexError('matricies should be 2x2, or kx2x2')
    return c


## COMMON CHECKS (raise exceptions)
def check_frequency_equal(ntwkA, ntwkB):
    '''
    checks if two Networks have same frequency
    '''
    if assert_frequency_equal(ntwkA,ntwkB) == False:
        raise IndexError('Networks dont have matching frequency. See `Network.interpolate`')

def check_z0_equal(ntwkA,ntwkB):
    '''
    checks if two Networks have same port impedances
    '''
    #note you should check frequency equal before you call this
    if assert_z0_equal(ntwkA,ntwkB) == False:
        raise ValueError('Networks dont have matching z0.')

def check_nports_equal(ntwkA,ntwkB):
    '''
    checks if two Networks have same number of ports
    '''
    if assert_nports_equal(ntwkA,ntwkB) == False:
        raise ValueError('Networks dont have matching number of ports.')
        
## TESTs (return [usually boolean] values)
def assert_frequency_equal(ntwkA, ntwkB):
    '''
    '''
    return (ntwkA.frequency  == ntwkB.frequency)

def assert_z0_equal(ntwkA,ntwkB):
    '''
    '''
    return (ntwkA.z0 == ntwkB.z0).all()

def assert_z0_at_ports_equal(ntwkA,k,ntwkB,l):
    '''
    '''
    return (ntwkA.z0[:,k] == ntwkB.z0[:,l]).all()

def assert_nports_equal(ntwkA,ntwkB):
    '''
    '''
    return (ntwkA.number_of_ports == ntwkB.number_of_ports)        




## Other
# dont belong here, but i needed them quickly
# this is needed for port impedance mismatches
def impedance_mismatch(z1, z2):
    '''
    creates a two-port s-matrix for a impedance mis-match

    Parameters
    -----------
    z1 : number or array-like
            complex impedance of port 1
    z2 : number or array-like
            complex impedance of port 2

    Returns
    ---------
    s' : 2-port s-matrix for the impedance mis-match
    '''
    gamma = zl_2_Gamma0(z1,z2)
    result = npy.zeros(shape=(len(gamma),2,2), dtype='complex')
    
    result[:,0,0] = gamma
    result[:,1,1] = -gamma
    result[:,1,0] = (1+gamma)*npy.sqrt(1.0*z1/z2)
    result[:,0,1] = (1-gamma)*npy.sqrt(1.0*z2/z1)
    return result

def two_port_reflect(ntwk1, ntwk2):
    '''
    generates a two-port reflective two-port, from two
    one-ports.


    Parameters
    ----------
    ntwk1 : one-port Network object
            network seen from port 1
    ntwk2 : one-port Network object
            network seen from port 2

    Returns
    -------
    result : Network object
            two-port reflective network

    Notes
    -------
        The resultant Network is copied from `ntwk1`, so its various 
    properties(name, frequency, etc) are inhereted from that Network.
    
    Examples
    ---------

    >>>short,open = rf.Network('short.s1p', rf.Network('open.s1p')
    >>>rf.two_port_reflect(short,open)
    '''
    result = ntwk1.copy()
    s11 = ntwk1.s[:,0,0]
    s22 = ntwk2.s[:,0,0]
    s21 = npy.zeros(ntwk1.frequency.npoints, dtype=complex)
    result.s = npy.array(\
            [[s11,  s21],\
            [ s21,  s22]]).\
            transpose().reshape(-1,2,2)
    result.z0 = npy.hstack([ntwk1.z0, ntwk2.z0])
    try:
        result.name = ntwk1.name+ntwk2.name
    except(TypeError):
        pass
    return result

