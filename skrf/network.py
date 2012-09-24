
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


Interpolation
================

.. autosummary::
    :toctree: generated/
    
    Network.interpolate
    Network.interpolate_self
    Network.interpolate_self_npoints (Network.resample)

   
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
    one_port_2_two_port
    impedance_mismatch
    load_all_touchstones
    write_dict_of_networks
    csv_2_touchstone



'''
import os
import warnings

import numpy as npy
import ctypes as ct     # for connect_s_fast
import pylab as plb
from scipy import stats         # for Network.add_noise_*
from scipy.interpolate import interp1d # for Network.interpolate()
import unittest # for unitest.skip 
import  mathFunctions as mf
import touchstone
from frequency import Frequency
from plotting import *#smith, plot_rectangular, plot_smith, plot_complex_polar
from tlineFunctions import zl_2_Gamma0

try:
    from src import connect_s_fast
except:
    pass#warnings.warn('libconnect failed to load.')


class Network(object):
    '''

    A n-port electrical network.

    A n-port network may be defined by three quantities,
     * scattering parameter matrix (s-matrix)
     * port characteristic impedance matrix
     * frequency information

    The :class:`Network` class stores these data structures internally
    in the form of complex numpy.ndarray's. These arrays are not
    interfaced directly but instead through the use of the properties:

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s`              scattering parameter matrix
    :attr:`z0`             characteristic impedance matrix
    :attr:`f`              frequency vector
    =====================  =============================================

    Individual components of the s-matrix are accesable through
    properties as well. These also return numpy.ndarray's.

    =====================  =============================================
    Property               Meaning
    =====================  =============================================
    :attr:`s_re`           real part of the s-matrix
    :attr:`s_im`           imaginary part of the s-matrix
    :attr:`s_mag`          magnitude of the s-matrix
    :attr:`s_db`           magnitude in log scale of the s-matrix
    :attr:`s_deg`          phase of the s-matrix in degrees
    =====================  =============================================

    The following :class:`Network` operators are available:

    =====================  =============================================
    Operator               Function
    =====================  =============================================
    \+                     element-wise addition of the s-matrix
    \-                     element-wise difference of the s-matrix
    \*                     element-wise multiplication of the s-matrix
    \/                     element-wise division of the s-matrix
    \*\*                     cascading (only for 2-ports)
    \//                    de-embedding (for 2-ports, see :attr:`inv`)
    =====================  =============================================

    Different components of the :class:`Network` can be visualized
    through various plotting methods. These methods can be used to plot
    individual elements of the s-matrix or all at once. For more info
    about plotting see the :doc:`../../tutorials/plotting` tutorial.

    =====================  =============================================
    Method                 Meaning
    =====================  =============================================
    :func:`plot_s_smith`   plot complex s-parameters on smith chart
    :func:`plot_s_re`      plot real part of s-parameters vs frequency
    :func:`plot_s_im`      plot imaginary part of s-parameters vs frequency
    :func:`plot_s_mag`     plot magnitude of s-parameters vs frequency
    :func:`plot_s_db`      plot magnitude (in dB) of s-parameters vs frequency
    :func:`plot_s_deg`     plot phase of s-parameters (in degrees) vs frequency
    =====================  =============================================

    Generally, :class:`Network`  objects are created from touchstone
    files upon initializtion  (see :func:`__init__`), or are created
    from a :class:`~media.media.Media` object. :class:`Network`  objects
    can be saved to disk in the form of touchstone files with the
    :func:`write_touchstone` method.

    An exhaustive list of :class:`Network` Methods and Properties
    (Attributes) are given below
    '''
    # used for testing s-parameter equivalence
    global ALMOST_ZER0
    ALMOST_ZER0=1e-6


    global PRIMARY_PROPERTIES
    PRIMARY_PROPERTIES = [ 's','z','y']

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
    def __init__(self, touchstone_file = None, name = None ):
        '''
        constructor.

        Contructs a Network, and optionally populates the s-matrix
        and frequency information from touchstone file.

        Parameters
        ------------

        file: string
                if given will load information from touchstone file,optional
        name: string
                name of this network, optional
        '''
        # although meaningless untill set with real values, this
        # needs this to exist for dependent properties
        #self.frequency = Frequency(0,0,0)

        if touchstone_file is not None:
            self.read_touchstone(touchstone_file)
            if name is not None:
                self.name = name
                

        else:
            self.name = name
            #self.s = None
            #self.z0 = 50

        #self.__generate_plot_functions()
        ##convenience
        self.resample = self.interpolate_self_npoints
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

    def __mul__(self,a):
        '''
        element-wise complex multiplication  of s-matrix
        '''
        self.__compatable_for_scalar_operation_test(other)
        result = self.copy()
        result.s = result.s * a.s
        return result

    def __add__(self,other):
        '''
        element-wise addition of s-matrix
        '''
        self.__compatable_for_scalar_operation_test(other)
        result = self.copy()
        result.s = result.s + other.s
        return result

    def __sub__(self,other):
        '''
        element-wise subtraction of s-matrix
        '''
        self.__compatable_for_scalar_operation_test(other)
        result = self.copy()
        result.s = result.s - other.s
        return result

    def __div__(self,other):
        '''
        element-wise division  of s-matrix
        '''
        self.__compatable_for_scalar_operation_test(other)
        result = self.copy()
        result.s =(self.s/ other.s)
        return result

    def __eq__(self,other):
        if npy.all(npy.abs(self.s - other.s) < ALMOST_ZER0):
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
                setattr(self.__class__,'%s_%s'%(prop_name, func_name),\
                    property(fget))

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
                setattr(self.__class__,'s%i%i'%(m+1,n+1),property(fget))

    ## PRIMARY PROPERTIES
    @property
    def s(self):
        '''
        the scattering parameter matrix [#]_.

        s-matrix is a 3 dimensional numpy.ndarray which has shape
        `fxnxn`, where `f` is frequency axis and `n` is number of ports.
        Note that indexing starts at 0!, so s11 can be accessed by 
        taking the slice s[:,0,0].  
        

        Returns
        ---------
        s : complex numpy.ndarry of shape `fxnxn`
                the scattering parameter matrix.

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
            elif len(s_shape) == 1:
                s = npy.reshape(s,(-1,1,1))

        self._s = s
        self.__generate_secondary_properties()
        self.__generate_subnetworks()
       
    @property
    def y(self):
        '''
        admittance parameters
        '''
        return s2y(self._s, self.z0)

    @y.setter
    def y(self, value):
        self._s = y2s(value, self.z0)
    
    @property
    def z(self):
        '''
        impedance parameters
        '''
        return s2z(self._s, self.z0)
    
    @z.setter
    def z(self, value):
        self._s = z2s(value, self.z0)
    
        
    @property
    def z0(self):
        '''
        the characteristic impedance[s] of the network ports.

        This property stores the  characteristic impedance of each port
        of the network. Because it is possible that each port has
        a different characteristic impedance, that is a function of
        frequency, `z0` is stored internally as a `fxn` array.

        However because frequently `z0` is simple (like 50ohm),it can
        be set with just number as well.

        Returns
        --------
        z0 : numpy.ndarray of shape fxn
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
        self._z0 = npy.array(z0)

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
    def t(self):
        '''
        t-parameters, aka scattering transfer parameters [#]_

        this is also known or the wave cascading matrix, and is only
        defined for a 2-port Network


        Returns
        --------
        t : complex numpy.ndarry of shape `fxnxn`
                t-parameters, aka scattering transfer parameters

        References
        -----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters
        '''
        return s2t(self.s)

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
        f : numpy.ndarray
                frequency vector in Hz

        See Also
        ---------
                frequency : frequency property that holds all frequency
                        information
        '''
        return self.frequency.f

    @f.setter
    def f(self,f):
        tmpUnit = self.frequency.unit
        self._frequency  = Frequency(f[0],f[-1],len(f),'hz')
        self._frequency.unit = tmpUnit

    
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
        passivity : numpy.ndarray of shape fxnxn

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
        returns a copy of this Network
        '''
        ntwk = Network()
        ntwk.frequency = self.frequency.copy()
        ntwk.s = self.s.copy()
        ntwk.z0 = self.z0.copy()
        ntwk.name = self.name
        return ntwk

    # touchstone file IO
    def read_touchstone(self, filename):
        '''
        loads values from a touchstone file.

        The work of this function is done through the
        :class:`~skrf.touchstone.touchstone` class.

        Parameters
        ----------
        filename : string
                touchstone file name.


        Notes
        ------
        only the scattering parameters format is supported at the
        moment



        '''
        touchstoneFile = touchstone.touchstone(filename)

        if touchstoneFile.get_format().split()[1] != 's':
            raise NotImplementedError('only s-parameters supported for now.')


        # set z0 before s so that y and z can be computed
        self.z0 = float(touchstoneFile.resistance)  
        self.f, self.s = touchstoneFile.get_sparameter_arrays() # note: freq in Hz
        self.frequency.unit = touchstoneFile.frequency_unit # for formatting plots
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

    def write_touchstone(self, filename=None, dir = './'):
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

        # write header file.
        # the '#'  line is NOT a comment it is essential and it must be
        #exactly this format, to work
        # [HZ/KHZ/MHZ/GHZ] [S/Y/Z/G/H] [MA/DB/RI] [R n]
        outputFile.write('!Created with skrf (http://scikit-rf.org).\n')
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

            outputFile.write('\n')
            outputFile.write('! Port Impedance\t' )
            for n in range(self.number_of_ports):
                outputFile.write('%.14f\t%.14f\t'%(self.z0[f,n].real, self.z0[f,n].imag))
            outputFile.write('\n')

        outputFile.close()

    
    # interpolation
    def interpolate(self, new_frequency,**kwargs):
        '''
        calculates an interpolated network.

        The default interpolation type is linear. see Notes for how to
        use other interpolation types.


        Parameters
        -----------
        new_frequency : :class:`~skrf.frequency.Frequency`
                frequency information to interpolate at
        **kwargs : keyword arguments
                passed to :func:`scipy.interpolate.interp1d` initializer.

        Returns
        ----------
        result : :class:`Network`
                an interpolated Network

        Notes
        --------
        useful keyword for  :func:`scipy.interpolate.interp1d`,
            kind : str or int
                Specifies the kind of interpolation as a string ('linear',
                'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or
                as an integer
                specifying the order of the spline interpolator to use.


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
        interpolate network based on a new number of frequency points

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
        '''
        new_frequency = self.frequency.copy()
        new_frequency.npoints = npoints
        self.interpolate_self(new_frequency, **kwargs)

    def interpolate_self(self, new_frequency, **kwargs):
        '''
        interpolates s-parameters given a new
        :class:'~skrf.frequency.Frequency' object.


        The default interpolation type is linear. see Notes for how to
        use other interpolation types.

        Parameters
        -----------
        new_frequency : :class:`~skrf.frequency.Frequency`
                frequency information to interpolate at
        **kwargs : keyword arguments
                passed to :func:`scipy.interpolate.interp1d` initializer.

        Notes
        --------
                useful keyword for  :func:`scipy.interpolate.interp1d`,
                 **kind** : str or int
                        Specifies the kind of interpolation as a string ('linear',
                        'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or
                        as an integer
                        specifying the order of the spline interpolator to use.

        See Also
        -----------
                interpolate : same function, but returns a new Network
        '''
        ntwk = self.interpolate(new_frequency, **kwargs)
        self.frequency, self.s,self.z0 = ntwk.frequency, ntwk.s,ntwk.z0

    def flip(self):
        '''
        swaps the ports of a two port Network
        '''
        if self.number_of_ports == 2:
            self.s = flip(self.s)
        else:
            raise ValueError('you can only flip two-port Networks')

    def renumber(self, from_ports, to_ports):
        '''
        renumbers some ports of a two port Network

        Parameters
        -----------
        from_ports : list-like
        to_ports: list-like
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
            chart_type='z', *args,**kwargs):
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
        *args : arguments, optional
                passed to the matplotlib.plot command
        **kwargs : keyword arguments, optional
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
                    smith(ax=ax, smithR = r, chart_type=chart_type)
                ax.plot(self.s[:,m,n].real,  self.s[:,m,n].imag, *args,**kwargs)

        #draw legend
        if show_legend:
            ax.legend()
        ax.axis(npy.array([-1,1,-1,1])*r)
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
        perturb s-parameters by small amount. this is useful to
        work-around numerical bugs.

        Parameters
        ------------
        amount : number,
                amount to add to s parameters

        '''
        self.s = self.s + 1e-12

## Functions operating on Network[s]
def connect(ntwkA, k, ntwkB,l):
    '''
    connect two n-port networks together.

    specifically, connect port `k` on `ntwkA` to port `l` on `ntwkB`. The
    resultant network has (ntwkA.nports+ntwkB.nports-2) ports. The port
    index's ('k','l') start from 0. Port impedances **are** taken into
    account.

    Parameters
    -----------
    ntwkA : :class:`Network`
            network 'A'
    k : int
            port index on `ntwkA` ( port indices start from 0 )
    ntwkB : :class:`Network`
            network 'B'
    l : int
            port index on `ntwkB`



    Returns
    ---------
    ntwkC : :class:`Network`
            new network of rank (ntwkA.nports + ntwkB.nports - 2*n)-ports


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
    # mismatch, which takes into account th effect of differing port
    # impedances. 
    #import pdb;pdb.set_trace()
    if assert_z0_at_ports_equal(ntwkA,k,ntwkB,l) == False:
        ntwkC.s = connect_s(
            ntwkA.s, k, 
            impedance_mismatch(ntwkA.z0[:,k], ntwkB.z0[:,l]), 0)
        ntwkC.z0[:,-1] = ntwkB.z0[:,l]
        ntwkC.renumber(from_ports=[k,-1], to_ports=[-1,k])

    # call s-matrix connection function
    ntwkC.s = connect_s(ntwkC.s,k,ntwkB.s,l)

    # remove rows and coloumns of z0 matrix which were `connected`
    ntwkC.z0 = npy.hstack(
        (npy.delete(ntwkA.z0, k, 1), npy.delete(ntwkB.z0, l, 1)))

    return ntwkC

def innerconnect(ntwkA, k, l):
    '''
    connect two ports of a single n-port network.

    this results in a (n-2)-port network. remember port indices start
    from 0.

    Parameters
    -----------
    ntwkA : :class:`Network`
        network 'A'
    k,l : int
        port indices on ntwkA ( port indices start from 0 )

    Returns
    ---------
    ntwkC : :class:`Network`
        new network of rank (ntwkA.nports+ntwkB.nports -2)-ports

    See Also
    -----------
        connect_s : actual  S-parameter connection algorithm.
        innerconnect_s : actual S-parameter connection algorithm.

    Notes
    -------
        a 2-port 'mismatch' network between the two connected ports.

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

    # call s-matrix connection function
    ntwkC.s = innerconnect_s(ntwkC.s,k,l)

    # update the characteristic impedance matrix
    ntwkC.z0 = npy.delete(ntwkC.z0,[l,k],1)

    return ntwkC

def cascade(ntwkA,ntwkB):
    '''
    cascade two 2-port Networks together

    connects port 1 of `ntwkA` to port 0 of `ntwkB`. This calls
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
    de-embed `ntwkA` from `ntwkB`. this calls `ntwkA.inv**ntwkB`.
    the syntax of cascading an inverse is more explicit, it is
    recomended that it be used instead of this function.

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

def average(list_of_networks):
    '''
    calculates the average network from a list of Networks.

    this is complex average of the s-parameters for a  list of Networks


    Parameters
    -----------
    list_of_networks: list
            a list of :class:`Network` objects

    Returns
    ---------
    ntwk : :class:`Network`
            the resultant averaged Network

    Notes
    ------
    This same function can be accomplished with properties of a
    :class:`NetworkSet` class.

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
    A : numpy.ndarray
            S-parameter matrix of `A`, shape is fxnxn
    k : int
            port index on `A` (port indices start from 0)
    B : numpy.ndarray
            S-parameter matrix of `B`, shape is fxnxn
    l : int
            port index on `B`

    Returns
    -------
    C : numpy.ndarray
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
    A : numpy.ndarray
        S-parameter matrix of `A`, shape is fxnxn
    k : int
        port index on `A` (port indices start from 0)
    l : int
        port index on `A`

    Returns
    -------
    C : numpy.ndarray
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
   
## network representation conversion       

def s2z(s,z0=50):
    '''
    convert scattering parameters to impedance parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
        
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
    '''
    if npy.isscalar(z0):
        z0 = npy.array(s.shape[0]*[s.shape[1] * [z0]])
    z = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    try:
        for fidx in xrange(s.shape[0]):
            sqrtz0 = npy.mat(npy.sqrt(npy.diagflat(z0[fidx])))
            z[fidx] = sqrtz0 * (I-s[fidx])**-1 * (I+s[fidx]) * sqrtz0
        return z
    except (npy.linalg.LinAlgError, ValueError):
        #print ('Warning: Cannot compute impedance parameters for network.')
        return None

def s2y(s,z0=50):
    '''
    convert scattering parameters to admittance parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    '''

    if npy.isscalar(z0):
        z0 = npy.array(s.shape[0]*[s.shape[1] * [z0]])
    y = npy.zeros(s.shape, dtype='complex')
    I = npy.mat(npy.identity(s.shape[1]))
    try:
        for fidx in xrange(s.shape[0]):
            sqrty0 = npy.mat(npy.sqrt(npy.diagflat(1.0/z0[fidx])))
            y[fidx] = sqrty0*(I-s[fidx])*(I+s[fidx])**-1*sqrty0
        return y
    except (npy.linalg.LinAlgError, ValueError):
        #print ('Warning: Cannot compute admittance parameters for network.')
        return None

def s2t(s):
    '''
    converts scattering parameters to scattering transfer parameters.

    transfer parameters [#]_ are also refered to as
    'wave cascading matrix', this function only operates on 2-port
    networks.

    Parameters
    -----------
    s : numpy.ndarray (shape fx2x2)
        scattering parameter matrix

    Returns
    -------
    t : numpy.ndarray
        scattering transfer parameters (aka wave cascading matrix)

    See Also
    ---------
    t2s : converts scattering transfer parameters to scattering
        parameters
    inv : calculates inverse s-parameters

    References
    -----------
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
    convert impedance parameters to scattering parameters [#]_

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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/impedance_parameters
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
    convert impedance parameters to admittance parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
    '''
    return npy.array([npy.mat(z[f,:,:])**-1 for f in xrange(z.shape[0])])
    

def z2t(z):
    '''
    Not Implemented yet
    
    convert impedance parameters to scattering transfer parameters [#]_
    

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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
    '''
    raise (NotImplementedError)

def y2s(y, z0=50):
    '''
    convert admittance parameters to scattering parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters

    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
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
    convert admittance parameters to impedance parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Admittance_parameters
    '''
    return npy.array([npy.mat(y[f,:,:])**-1 for f in xrange(y.shape[0])])

def y2t(y):
    '''
    Not Implemented Yet 
    
    convert admittance parameters to scattering-transfer parameters [#]_


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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
    '''
    raise (NotImplementedError)

def t2s(t):
    '''
    converts scattering transfer parameters to scattering parameters

    transfer parameters [#]_ are also refered to as
    'wave cascading matrix', this function only operates on 2-port
    networks. this function only operates on 2-port scattering
    parameters.

    Parameters
    -----------
    t : numpy.ndarray (shape fx2x2)
            scattering transfer parameters

    Returns
    -------
    s : numpy.ndarray
            scattering parameter matrix.

    See Also
    ---------
    t2s : converts scattering transfer parameters to scattering parameters
    inv : calculates inverse s-parameters

    References
    -----------
    .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
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
    
    Convert scattering transfer parameters to impedance parameters [#]_



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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
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
    s2z : converts scattering parameters to impedance parameters
    s2y : converts scattering parameters to admittance parameters
    s2t : converts scattering parameters to scattering transfer
        parameters
    z2s : converts impedance parameters to scattering parameters
    z2y : converts impedance parameters to impedance parameters
    z2t : converts impedance parameters to scattering transfer
        parameters
    y2s : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to impedance parameters
    y2z : converts admittance parameters to scattering transfer
        parameters
    t2s : converts scattering transfer paramerters to scattering
        parameters
    t2z : converts scattering transfer paramerters to impedance
        parameters
    t2y : converts scattering transfer paramerters to admittance
        parameters
    
    References
    ----------
    .. [#] http://en.wikipedia.org/wiki/Two-port_network
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
    s : numpy.ndarray (shape fx2x2)
            scattering parameter matrix.

    Returns
    -------
    s' : numpy.ndarray
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
    a : numpy.ndarray
            scattering parameter matrix. shape should be should be 2x2, or
            fx2x2

    Returns
    -------
    a' : numpy.ndarray
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
# TODO: would like to nose from running these, but i dont know how
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

# Touchstone manipulation
def load_all_touchstones(dir = '.', contains=None, f_unit=None):
    '''
    loads all touchtone files in a given dir into a dictionary.

    Parameters
    -----------
    dir :   string
            the path
    contains :      string
            a string the filenames must contain to be loaded.
    f_unit  : ['hz','mhz','ghz']
            the frequency unit to assign all loaded networks. see
            :attr:`frequency.Frequency.unit`.

    Returns
    ---------
    ntwkDict : a dictonary with keys equal to the file name (without
            a suffix), and values equal to the corresponding ntwk types

    Examples
    ----------
    >>> ntwk_dict = rf.load_all_touchstones('.', contains ='20v')

    '''
    ntwkDict = {}

    for f in os.listdir (dir):
        if contains is not None and contains not in f:
            continue

        # TODO: make this s?p with reg ex
        if( f.lower().endswith ('.s1p') or f.lower().endswith ('.s2p') ):
            name = f[:-4]
            ntwkDict[name]=(Network(dir +'/'+f))
            if f_unit is not None: ntwkDict[name].frequency.unit=f_unit

    return ntwkDict

def write_dict_of_networks(ntwkDict, dir='.'):
    '''
    saves a dictionary of networks touchstone files in a given directory

    The filenames assigned to the touchstone files are taken from
    the keys of the dictionary.

    Parameters
    -----------
    ntwkDict : dictionary
            dictionary of :class:`Network` objects
    dir : string
            directory to write touchstone file to


    '''
    for ntwkKey in ntwkDict:
        ntwkDict[ntwkKey].write_touchstone(filename = dir+'/'+ntwkKey)

def csv_2_touchstone(filename):
    '''
    converts a csv file to a :class:`Network`

    specifically, this converts csv files saved from a Rohde Shcwarz
    ZVA-40, and possibly other network analyzers, into a :class:`Network`
    object.

    Parameters
    ------------
    filename : string
            name of file

    Returns
    --------
    ntwk : :class:`Network` object
            the network representing data in the csv file
    '''

    ntwk = Network(name=filename[:-4])
    try:
        data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(9))
        s11 = data[:,1] +1j*data[:,2]
        s21 = data[:,3] +1j*data[:,4]
        s12 = data[:,5] +1j*data[:,6]
        s22 = data[:,7] +1j*data[:,8]
        ntwk.s = npy.array([[s11, s21],[s12,s22]]).transpose().reshape(-1,2,2)
    except(IndexError):
        data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(3))
        ntwk.s = data[:,1] +1j*data[:,2]

    ntwk.frequency.f = data[:,0]
    ntwk.frequency.unit='ghz'

    return ntwk

