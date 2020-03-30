
'''
.. module:: skrf.media.media
========================================
media (:mod:`skrf.media.media`)
========================================

Contains Media class.


'''
import warnings

import numpy as npy
from numpy import real, imag, ones, any, gradient, array
from scipy import stats
from scipy.constants import  c, inch, mil

from ..frequency import Frequency
from ..network import Network, connect

from .. import tlineFunctions as tf
from .. import mathFunctions as mf

from ..constants import to_meters ,ZERO

from abc import ABCMeta, abstractmethod, abstractproperty
import re
from copy import deepcopy as copy

class Media(object):
    '''
    Abstract Base Class for a single mode on a transmission line media.
    
    
    This class init's with `frequency` and `z0` (the port impedance);
    attributes shared by all media. Methods defined here make use of the 
    properties :
    
    * `gamma` - (complex) propgation constant
    * `Z0` - (complex) characteristic impedance
    
    Which define the properties of specific media. Any sub-class of Media 
    must implement these properties. `gamma` and `Z0` should return 
    complex arrays of the same length as `frequency`. `gamma` must 
    follow the convention,
    
    * positive real(gamma) = attenuation
    * positive imag(gamma) = forward propagation
    
    Parameters
    --------------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of this transmission line medium. If None, will 
        default to  1-10ghz, 


    z0 : number, array-like, or None
        the port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if z0 is None then will default to Z0


    Notes
    --------
    The z0 parameter is needed in some cases. 
    :class:`~skrf.media.rectangularWaveguide.RectangularWaveguide`
    is an example  where you may need this, because the
    characteristic impedance is frequency dependent, but the
    touchstone's created by most VNA's have z0=1, or 50. so to 
    prevent accidental impedance mis-match, you may want to manually
    set the z0 .
    '''
    __metaclass__ = ABCMeta
    def __init__(self, frequency=None, z0=None):
        if frequency is None:
            frequency = Frequency(1,10,101,'ghz')
        
        self.frequency = frequency.copy()
        self.z0 = z0

    def mode(self,  **kw):
        '''
        create another mode in this medium 
        
        convenient way to copy this media object, with 
        '''
        out = copy(self)
        for k in kw:
            setattr(self, k, kw[k])
        return out

    def copy(self):
        return copy(self)
            
    def __eq__(self,other):
        '''
        test for numerical equality (up to skrf.constants.ZERO)
        '''

        if self.frequency != other.frequency:
            return False

        if max(abs(self.Z0 - other.Z0)) > ZERO:
            return False

        if max(abs(self.gamma - other.gamma)) > ZERO:
            return False

        if max(abs(self.z0 - other.z0)) > ZERO:
            return False

        return True

    def __len__(self):
        '''
        length of frequency axis
        '''
        return len(self.frequency)

    @property
    def npoints(self):
        return self.frequency.npoints
    @npoints.setter
    def npoints(self,val):
        self.frequency.npoints = val
    
    @property
    def z0(self):
        if self._z0 is None:
            return self.Z0
        return self._z0*ones(len(self))
        
    @z0.setter
    def z0(self, val):
        self._z0 = val

    @abstractproperty
    def gamma(self):
        '''
        Propagation constant

        Returns
        ---------
        gamma : :class:`numpy.ndarray`
                complex propagation constant for this media

        Notes
        ------
        `gamma` must adhere to the following convention,
         * positive real(gamma) = attenuation
         * positive imag(gamma) = forward propagation
        '''
        return None
    
    @gamma.setter
    def gamma(self, val):
        pass
        
    @property
    def alpha(self):
        '''
        real (attenuation) component of gamma
        '''
        return real(self.gamma)
        
    @property
    def beta(self):
        '''
        imaginary (propagating) component of gamma
        '''
        return imag(self.gamma)
    
    @abstractproperty
    def Z0(self):
        return None
    
    @Z0.setter
    def Z0(self, val):
        pass 
        
    

    @property
    def v_p(self):
        '''
        Complex phase velocity (in m/s)

        .. math::
            j \cdot \\omega / \\gamma

        Notes
        -------
        The `j` is used so that real phase velocity corresponds to
        propagation

        where:
        * :math:`\\omega` is angular frequency (rad/s),
        * :math:`\\gamma` is complex propagation constant (rad/m)

        See Also
        -----------
        propgation_constant

        '''
        return 1j*(self.frequency.w/self.gamma)


    @property
    def v_g(self):
        '''
        Complex group velocity (in m/s)

        .. math::
            j \cdot d \\omega / d \\gamma


        where:
        * :math:`\\omega` is angular frequency (rad/s),
        * :math:`\\gamma` is complex propagation constant (rad/m)
    
        Notes
        -----
        the `j` is used to make propgation real, this is needed  because 
        skrf defined the gamma as \\gamma= \\alpha +j\\beta.
        
    
        References 
        -------------
        https://en.wikipedia.org/wiki/Group_velocity

        See Also
        -----------
        propgation_constant
        v_p
        '''
        dw = self.frequency.dw
        dk = gradient(self.gamma)

        return dw/dk


    def get_array_of(self,x):
        try:
            if len(x)!= len(self):
                # we have to make a decision
                pass
        except(TypeError):
            y = x* ones(len(self))
        
        return y

    ## Other Functions
    def theta_2_d(self,theta,deg=True, bc = True):
        '''
        Converts electrical length to physical distance.

        The given electrical length is to be  at the center frequency.

        Parameters
        ----------
        theta : number
                electrical length, at band center (see deg for unit)
        deg : Boolean
                is theta in degrees?

        bc : bool
                evaluate only at band center, or across the entire band?

        Returns
        --------
        d : number, array-like
                physical distance in meters


        '''
        if deg == True:
            theta = mf.degree_2_radian(theta)

        gamma = self.gamma
        if bc:
                return 1.0*theta/npy.imag(gamma[int(gamma.size/2)])
        else:
                return 1.0*theta/npy.imag(gamma)

    def electrical_length(self, d,deg=False):
        '''
        calculates the electrical length for a given distance


        Parameters
        ----------
        d: number or array-like
            delay distance, in meters

        deg: Boolean
            return electral length in deg?

        Returns
        --------
        theta: number or array-like
            electrical length in radians or degrees, depending on
            value of deg.
        '''
        gamma = self.gamma

        if deg == False:
            return  gamma*d
        elif deg == True:
            return  mf.radian_2_degree(gamma*d )

    ## Network creation

    # lumped elements
    def match(self,nports=1, z0=None, z0_norm=False, **kwargs):
        '''
        Perfect matched load (:math:`\\Gamma_0 = 0`).

        Parameters
        ----------
        nports : int
                number of ports
        z0 : number, or array-like
                port impedance. Default is
                None, in which case the Media's :attr:`z0` is used.
                This sets the resultant Network's
                :attr:`~skrf.network.Network.z0`.
        z0_norm :bool
            is z0  normalized to this media's `z0`?
        \*\*kwargs : key word arguments
                passed to :class:`~skrf.network.Network` initializer

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port match


        Examples
        ------------
                >>> my_match = my_media.match(2,z0 = 50, name='Super Awesome Match')

        '''
        
        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s =  npy.zeros((self.frequency.npoints,nports, nports),\
                dtype=complex)
        if z0 is None:
            z0 = self.z0
        elif isinstance(z0,str):
            z0 = parse_z0(z0)* self.z0
            
        if z0_norm:
            z0 = z0*self.z0
        
        result.z0=z0
        return result

    def load(self,Gamma0,nports=1,**kwargs):
        '''
        Load of given reflection coefficient.

        Parameters
        ----------
        Gamma0 : number, array-like
                Reflection coefficient of load (linear, not in db). If its
                an array it must be of shape: kxnxn, where k is #frequency
                points in media, and n is `nports`
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        load  :class:`~skrf.network.Network` object
                n-port load, where  S = Gamma0*eye(...)
        '''
        result = self.match(nports,**kwargs)
        result.s =  npy.array(Gamma0).reshape(-1,1,1)* \
                    npy.eye(nports,dtype=complex).reshape((-1,nports,nports)).\
                    repeat(self.frequency.npoints,0)
        #except(ValueError):
        #    for f in range(self.frequency.npoints):
        #        result.s[f,:,:] = Gamma0[f]*npy.eye(nports, dtype=complex)

        return result

    def short(self,nports=1,**kwargs):
        '''
        Short (:math:`\\Gamma_0 = -1`)

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port short circuit

        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        return self.load(-1., nports, **kwargs)

    def open(self,nports=1, **kwargs):
        '''
        Open (:math:`\\Gamma_0 = 1`)

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        match : :class:`~skrf.network.Network` object
                a n-port open circuit

        See Also
        ---------
        match : function called to create a 'blank' network
        '''

        return self.load(1., nports, **kwargs)

    def resistor(self, R, *args, **kwargs):
        '''
        Resistor


        Parameters
        ----------
        R : number, array
                Resistance , in Ohms. If this is an array, must be of
                same length as frequency vector.
        \*args, \*\*kwargs : arguments, key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        resistor : a 2-port :class:`~skrf.network.Network`

        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        result = self.match(nports=2, *args, **kwargs)
        y= npy.zeros(shape=result.s.shape, dtype=complex)
        y[:,0,0] = 1./R
        y[:,1,1] = 1./R
        y[:,0,1] = -1./R
        y[:,1,0] = -1./R
        result.y = y
        return result

    def capacitor(self, C, **kwargs):
        '''
        Capacitor


        Parameters
        ----------
        C : number, array
                Capacitance, in Farads. If this is an array, must be of
                same length as frequency vector.
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        capacitor : a 2-port :class:`~skrf.network.Network`


        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        result = self.match(nports=2, **kwargs)
        w = self.frequency.w
        y= npy.zeros(shape=result.s.shape, dtype=complex)
        y[:,0,0] = 1j*w*C
        y[:,1,1] = 1j*w*C
        y[:,0,1] = -1j*w*C
        y[:,1,0] = -1j*w*C
        result.y = y
        return result

    def inductor(self, L, **kwargs):
        '''
        Inductor

        Parameters
        ----------
        L : number, array
                Inductance, in Henrys. If this is an array, must be of
                same length as frequency vector.
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        inductor : a 2-port :class:`~skrf.network.Network`


        See Also
        ---------
        match : function called to create a 'blank' network
        '''
        result = self.match(nports=2, **kwargs)
        w = self.frequency.w
        y = npy.zeros(shape=result.s.shape, dtype=complex)
        y[:,0,0] = 1./(1j*w*L)
        y[:,1,1] = 1./(1j*w*L)
        y[:,0,1] = -1./(1j*w*L)
        y[:,1,0] = -1./(1j*w*L)
        result.y = y
        return result

    def impedance_mismatch(self, z1, z2, **kwargs):
        '''
        Two-port network for an impedance mismatch


        Parameters
        ----------
        z1 : number, or array-like
                complex impedance of port 1
        z2 : number, or array-like
                complex impedance of port 2
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        missmatch : :class:`~skrf.network.Network` object
                a 2-port network representing the impedance mismatch

        Notes
        --------
        If z1 and z2 are arrays, they must be of same length
        as the :attr:`Media.frequency.npoints`

        See Also
        ---------
        match : called to create a 'blank' network
        '''
        result = self.match(nports=2, **kwargs)
        gamma = tf.zl_2_Gamma0(z1,z2)
        result.s[:,0,0] = gamma
        result.s[:,1,1] = -gamma
        result.s[:,1,0] = (1+gamma)*npy.sqrt(1.0*z1/z2)
        result.s[:,0,1] = (1-gamma)*npy.sqrt(1.0*z2/z1)
        return result


    # splitter/couplers
    def tee(self,**kwargs):
        '''
        Ideal, lossless tee. (3-port splitter)

        Parameters
        ----------
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        tee : :class:`~skrf.network.Network` object
                a 3-port splitter

        See Also
        ----------
        splitter : this just calls splitter(3)
        match : called to create a 'blank' network
        '''
        return self.splitter(3,**kwargs)

    def splitter(self, nports,**kwargs):
        '''
        Ideal, lossless n-way splitter.

        Parameters
        ----------
        nports : int
                number of ports
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        tee : :class:`~skrf.network.Network` object
                a n-port splitter

        See Also
        ---------
        match : called to create a 'blank' network
        '''
        n=nports
        result = self.match(n, **kwargs)

        for f in range(self.frequency.npoints):
            result.s[f,:,:] =  (2*1./n-1)*npy.eye(n) + \
                    npy.sqrt((1-((2.-n)/n)**2)/(n-1))*\
                    (npy.ones((n,n))-npy.eye(n))
        return result


    # transmission line

    def to_meters(self, d, unit='deg'):
        '''
        Translate various  units of distance into meters

        This is a method of media to allow for electrical lengths as
        inputs.  For dispersive media, mean group velocity is used to 
        translate time-based units to distance.

        Parameters
        ------------
        d : number or array-like
            the value
        unit : str
            the unit to that x is in:
            ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']

        See Also
        ----------
        skrf.constants.to_meters
        '''
        unit = unit.lower()
        #import pdb;pdb.set_trace()
        
        d_dict ={'deg':self.theta_2_d(d,deg=True),
                 'rad':self.theta_2_d(d,deg=False),
                 }
        
        if unit in d_dict: 
            return d_dict[unit]
        else:
            # mean group velocity is used to translate time-based
            # units to distance
            if 's' in unit:
                # they are specifiying  a time unit so calculate
                # the group velocity. (note this fails for media of 
                # too little points, as it uses gradient)
                v_g = -self.v_g.imag.mean()
            else:
                v_g=c
            return to_meters(d=d,unit=unit, v_g=v_g)

    def thru(self, **kwargs):
        '''
        Matched transmission line of length 0.

        Parameters
        ----------
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        thru : :class:`~skrf.network.Network` object
                matched tranmission line of 0 length

        See Also
        ---------
        line : this just calls line(0)
        '''
        return self.line(0,**kwargs)

    def line(self,d, unit='deg',z0=None, embed = False, **kwargs):
        '''
        Transmission line of a given length and impedance

        The units of `length` are interpreted according to the value
        of `unit`. If `z0` is not None, then a line specified  impedance
        is produced. if `embed`  is also True, then the line is embedded
        in this media's z0 environment, creating a mismatched line.

        Parameters
        ----------
        d : number
                the length of transmissin line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        z0 : number, string, or array-like
                the characteristic impedance of the line, if different 
                from self.z0. To set z0 in terms of normalized impedance,
                pass a string, like `z0='1+.2j'`
                
        embed : bool
                if `Z0` is given, should the line be embedded in z0
                environment? or left in a `z` environment. if embedded,
                there will be reflections
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        line : :class:`~skrf.network.Network` object
                matched tranmission line of given length

        Examples
        ----------
        >>> my_media.line(1, 'mm', z0=100)
        >>> my_media.line(90,'deg',z0='2') # set z0 as normalized impedance

        '''
        
        if isinstance(z0,str):
            z0 = parse_z0(z0)* self.z0
            
        kwargs.update({'z0':z0})
        result = self.match(nports=2,**kwargs)

        theta = self.electrical_length(self.to_meters(d=d, unit=unit))

        s11 = npy.zeros(self.frequency.npoints, dtype=complex)
        s21 = npy.exp(-1*theta)
        result.s = \
                npy.array([[s11, s21],[s21,s11]]).transpose().reshape(-1,2,2)

        if  embed:
            result = self.thru()**result**self.thru()

        return result


    def delay_load(self,Gamma0,d,unit='deg',**kwargs):
        '''
        Delayed load

        A load with reflection coefficient `Gamma0` at the end of a
        matched line of length `d`.

        Parameters
        ----------
        Gamma0 : number, array-like
                reflection coefficient of load (not in dB)
        d : number
                the length of transmissin line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_load : :class:`~skrf.network.Network` object
                a delayed load


        Examples
        ----------
        >>> my_media.delay_load(-.5, 90, 'deg', Z0=50)


        Notes
        ------
        This calls ::

                line(d,unit, **kwargs) ** load(Gamma0, **kwargs)

        See Also
        ---------
        line : creates the network for line
        load : creates the network for the load


        '''
        return self.line(d=d, unit=unit,**kwargs)**\
                self.load(Gamma0=Gamma0,**kwargs)

    def delay_short(self,d,unit='deg',**kwargs):
        '''
        Delayed Short

        A transmission line of given length terminated with a short.

        Parameters
        ----------
        d : number
                the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_short : :class:`~skrf.network.Network` object
                a delayed short


        See Also
        --------
        delay_load : delay_short just calls this function

        '''
        return self.delay_load(Gamma0=-1., d=d, unit=unit, **kwargs)

    def delay_open(self,d,unit='deg',**kwargs):
        '''
        Delayed open transmission line

        Parameters
        ----------
        d : number
                the length of transmission line (see unit argument)
        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details
        \*\*kwargs : key word arguments
                passed to :func:`match`, which is called initially to create a
                'blank' network.

        Returns
        --------
        delay_open : :class:`~skrf.network.Network` object
                a delayed open


        See Also
        ---------
        delay_load : delay_short just calls this function
        '''
        return self.delay_load(Gamma0=1., d=d, unit=unit,**kwargs)

    def shunt(self,ntwk, **kwargs):
        '''
        Shunts a :class:`~skrf.network.Network`

        This creates a :func:`tee` and connects connects
        `ntwk` to port 1, and returns the result

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
        \*\*kwargs : keyword arguments
                passed to :func:`tee`

        Returns
        --------
        shunted_ntwk : :class:`~skrf.network.Network` object
                a shunted a ntwk. The resultant shunted_ntwk will have
                (2 + ntwk.number_of_ports -1) ports.

        '''
        return connect(self.tee(**kwargs),1,ntwk,0)

    def shunt_delay_load(self,*args, **kwargs):
        '''
        Shunted delayed load

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_load`

        Returns
        --------
        shunt_delay_load : :class:`~skrf.network.Network` object
                a shunted delayed load (2-port)

        Notes
        --------
        This calls::

                shunt(delay_load(*args, **kwargs))

        '''
        return self.shunt(self.delay_load(*args, **kwargs))

    def shunt_delay_open(self,*args,**kwargs):
        '''
        Shunted delayed open

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_delay_open : :class:`~skrf.network.Network` object
                shunted delayed open (2-port)

        Notes
        --------
        This calls::

                shunt(delay_open(*args, **kwargs))
        '''
        return self.shunt(self.delay_open(*args, **kwargs))

    def shunt_delay_short(self,*args,**kwargs):
        '''
        Shunted delayed short

        Parameters
        ----------
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_delay_load : :class:`~skrf.network.Network` object
                shunted delayed open (2-port)

        Notes
        --------
        This calls::

                shunt(delay_short(*args, **kwargs))
        '''
        return self.shunt(self.delay_short(*args, **kwargs))

    def shunt_capacitor(self,C,*args,**kwargs):
        '''
        Shunted capacitor

        Parameters
        ----------
        C : number, array-like
                Capacitance in Farads.
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_capacitor : :class:`~skrf.network.Network` object
                shunted capcitor(2-port)

        Notes
        --------
        This calls::

                shunt(capacitor(C,*args, **kwargs))

        '''
        return self.shunt(self.capacitor(C=C,*args,**kwargs)**self.short())

    def shunt_inductor(self,L,*args,**kwargs):
        '''
        Shunted inductor

        Parameters
        ----------
        L : number, array-like
                Inductance in Farads.
        \*args,\*\*kwargs : arguments, keyword arguments
                passed to func:`delay_open`

        Returns
        --------
        shunt_inductor : :class:`~skrf.network.Network` object
                shunted inductor(2-port)

        Notes
        --------
        This calls::

                shunt(inductor(C,*args, **kwargs))

        '''
        return self.shunt(self.inductor(L=L,*args,**kwargs)**self.short())

    def attenuator(self, s21, db=True, d =0, unit='deg', name='',**kwargs):
        '''
        Ideal matched attenuator of a given length

        Parameters
        ----------
        s21 : number, array-like
            the attenutation
        db : bool
            is s21 in db? otherwise assumes linear
        d : number
            length of attenuator

        unit : ['deg','rad','m','cm','um','in','mil','s','us','ns','ps']
                the units of d.  See :func:`to_meters`, for details

        Returns
        --------
        ntwk : :class:`~skrf.network.Network` object
                2-port attentuator

        '''
        if db:
            s21 = mf.db_2_magnitude(s21)

        result = self.match(nports=2)
        result.s[:,0,1] = s21
        result.s[:,1,0] = s21
        result = result**self.line(d=d, unit = unit, **kwargs)
        result.name = name
        return result

    def lossless_mismatch(self,s11,db=True,  **kwargs):
        '''
        Lossless, symmetric mismatch defined by its return loss

        Parameters
        ----------
        s11 : complex number, number, or array-like
            the reflection coefficient. if db==True, then phase is ignored

        db : bool
            is s11 in db? otherwise assumes linear

        Returns
        --------
        ntwk : :class:`~skrf.network.Network` object
                2-port lossless mismatch

        '''
        result = self.match(nports=2,**kwargs)
        if db:
            s11 = mf.db_2_magnitude(s11)

        result.s[:,0,0] = s11
        result.s[:,1,1] = s11

        s21_mag = npy.sqrt(1- npy.abs(s11)**2)
        s21_phase = (npy.angle(s11) \
                   + npy.pi/2 *(npy.angle(s11)<=0) \
                   - npy.pi/2 *(npy.angle(s11)>0))
        result.s[:,0,1] =  s21_mag* npy.exp(1j*s21_phase)
        result.s[:,1,0] = result.s[:,0,1]
        return result
    
    def isolator(self,source_port=0,**kwargs):
        '''
        two-port isolator 
        
        
        Parameters
        -------------
        source_port: [0,1]
            port at which power can flow from.
        '''
        result = self.thru(**kwargs)
        if source_port==0:
            result.s[:,0,1]=0
        elif source_port==1:
            result.s[:,1,0]=0
        return result
            
        
    
    ## Noise Networks
    def white_gaussian_polar(self,phase_dev, mag_dev,n_ports=1,**kwargs):
        '''
        Complex zero-mean gaussian white-noise network.

        Creates a network whose s-matrix is complex zero-mean gaussian
        white-noise, of given standard deviations for phase and
        magnitude components.
        This 'noise' network can be added to networks to simulate
        additive noise.

        Parameters
        ----------
        phase_mag : number
                standard deviation of magnitude
        phase_dev : number
                standard deviation of phase
        n_ports : int
                number of ports.
        \*\*kwargs : passed to :class:`~skrf.network.Network`
                initializer

        Returns
        --------
        result : :class:`~skrf.network.Network` object
                a noise network
        '''
        shape = (self.frequency.npoints, n_ports,n_ports)
        phase_rv= stats.norm(loc=0, scale=phase_dev).rvs(size = shape)
        mag_rv = stats.norm(loc=0, scale=mag_dev).rvs(size = shape)

        result = Network(**kwargs)
        result.frequency = self.frequency
        result.s = mag_rv*npy.exp(1j*phase_rv)
        return result

    def random(self, n_ports=1, reciprocal=False, matched=False, 
               symmetric=False, **kwargs):
        '''
        Complex random network.

        Creates a n-port network whose s-matrix is filled with random
        complex numbers. Optionaly, result can be matched or reciprocal.

        Parameters
        ----------
        n_ports : int
            number of ports.
        reciprocal : bool
            makes s-matrix symmetric ($S_{mn} = S_{nm}$)
        symmetric : bool
            makes s-matrix diagonal have single value ($S_{mm}=S_{nn}$)
        matched : bool
            makes diagonals of s-matrix zero

        \*\*kwargs : passed to :class:`~skrf.network.Network`
                initializer

        Returns
        --------
        result : :class:`~skrf.network.Network` object
                the network
        '''
        result = self.match(nports = n_ports, **kwargs)
        result.s = mf.rand_c(self.frequency.npoints, n_ports,n_ports)
        if reciprocal and n_ports>1:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m>n:
                        result.s[:,m,n] = result.s[:,n,m]
        if symmetric:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m==n:
                        result.s[:,m,n] = result.s[:,0,0] 
        if matched:
            for m in range(n_ports):
                for n in range(n_ports):
                    if m==n:
                        result.s[:,m,n] = 0

        return result

    ## OTHER METHODS
    def extract_distance(self,ntwk):
        '''
        Determines physical distance from a transmission or reflection ntwk
        
        Given a matched transmission or reflection measurment the 
        physical distance is estimated at each frequency point based on 
        the scattering parameter phase of the ntwk and propagation constant.

        Notes
        -------
        If the ntwk is a reflect measurement, the returned distance will  
        be twice the physical distance.
        
        Parameters
        -----------
        ntwk : `Network`
            A one-port network of either the reflection or the transmission.
            if
        
        Example
        ----------
        >>>air = rf.air50
        >>>l = air.line(1,'cm')
        >>>d_found = air.extract_distance(l.s21) 
        >>>d_found
        '''
        if ntwk.nports ==1:
            dphi = gradient(ntwk.s_rad_unwrap.flatten())
            dgamma = gradient(self.gamma.imag)
            return  -dphi/dgamma
        else:
            raise ValueError('ntwk must be one-port. Select s21 or s12 for a two-port.')
    


    def plot(self, *args, **kw):
        return self.frequency.plot(*args, **kw)

    

    def write_csv(self, filename='f,gamma,Z0,z0.csv'):
        '''
        write this media's frequency,gamma,Z0, and z0 to a csv file.

        Parameters
        -------------
        filename : string
            file name to write out data to

        See Also
        ---------
        from_csv : class method to initialize Media object from a
            csv file written from this function
        '''

        header = 'f[%s], Re(Z0), Im(Z0), Re(gamma), Im(gamma), Re(port Z0), Im(port Z0)\n'%self.frequency.unit
        
        g,z,pz  = self.gamma, \
                self.Z0, self.z0

        data = npy.vstack(\
                [self.frequency.f_scaled, z.real, z.imag, \
                g.real, g.imag, pz.real, pz.imag]).T

        npy.savetxt(filename,data,delimiter=',',header=header)



class DefinedGammaZ0(Media):
    '''
    A media directly defined by its propagation constant and 
    characteristic impedance
    
    Parameters
    --------------
    frequency : :class:`~skrf.frequency.Frequency` object
        frequency band of this transmission line medium. If None, will 
        default to  1-10ghz, 


    z0 : number, array-like, or None
        The port impedance for media. Only needed if  its different
        from the characterisitc impedance of the transmission
        line. if `z0` is `None` then it will default to `Z0`
 
    gamma : number, array-like
        complex propagation constant. `gamma` must adhere to 
        the following convention,
            * positive real(gamma) = attenuation
            * positive imag(gamma) = forward propagation
        
    Z0 : number, array-like
        complex characteristic impedance.
    '''
    def __init__(self, frequency=None, z0=None, gamma=1j, Z0=50):
        '''
        '''
        super(DefinedGammaZ0, self).__init__(frequency=frequency, 
                                             z0=z0)
        self.gamma= gamma
        self.Z0 = Z0
    
    @classmethod
    def from_csv(cls, filename, *args, **kwargs):
        '''
        create a Media from numerical values stored in a csv file.

        the csv file format must be written by the function write_csv(),
        or similar method  which produces the following format

            f[$unit], Re(Z0), Im(Z0), Re(gamma), Im(gamma), Re(port Z0), Im(port Z0)
            1, 1, 1, 1, 1, 1, 1
            2, 1, 1, 1, 1, 1, 1
            .....

        '''
        try:
            f = open(filename)
        except(TypeError):
            # they may have passed a file
            f = filename

        header = f.readline()
        # this is not the correct way to do this ... but whatever
        f_unit = header.split(',')[0].split('[')[1].split(']')[0]

        f,z_re,z_im,g_re,g_im,pz_re,pz_im = \
                npy.loadtxt(f,  delimiter=',').T

        return cls(
            frequency = Frequency.from_f(f, unit=f_unit),
            Z0 = z_re+1j*z_im,
            gamma = g_re+1j*g_im,
            z0 = pz_re+1j*pz_im,
            *args, **kwargs
            )
    @property
    def npoints(self):
        return self.frequency.npoints
    
    @npoints.setter
    def npoints(self,val):
        # this is done to trigger checks on vector lengths for 
        # gamma/Z0/z0
        new_freq= self.frequency.copy()
        new_freq.npoints = val
        self.frequency = new_freq
        
    
    @property
    def frequency(self):
        return self._frequency
        
    @frequency.setter
    def frequency(self, val):
        if hasattr(self, '_frequency') and self._frequency is not None:
            
            # they are updating the frequency, we may have to do somethign
            attrs_to_test = [self._gamma, self._Z0, self._z0]
            if any([has_len(k) for k in attrs_to_test]):
                 raise NotImplementedError('updating a Media frequency, with non-constant gamma/Z0/z0 is not worked out yet')
        self._frequency = val
        
    @property
    def Z0(self):
        '''
        Characteristic Impedance 
        '''
        return self._Z0*ones(len(self))
    
    @Z0.setter
    def Z0(self, val):
        self._Z0 = val
    
    @property
    def gamma(self):
        '''
        Propagation constant

        Returns
        ---------
        gamma : :class:`numpy.ndarray`
            complex propagation constant for this media

        Notes
        ------
        `gamma` must adhere to the following convention,
         * positive real(gamma) = attenuation
         * positive imag(gamma) = forward propagation
        '''
        return self._gamma*ones(len(self))
    
    @gamma.setter
    def gamma(self, val):
        self._gamma = val

def has_len(x):
    '''
    test of x  has any length  (ie is a vector)
    
    this is slightly non-trivial because [3] has len() but is 
    doesnt really have any length
    '''
    try:
        return (len(array(x))>1)
    except TypeError:
        return False

def parse_z0(s):
    # they passed a string for z0, try to parse it 
    re_numbers = re.compile('\d+')
    numbers = re.findall(re_numbers, s)
    if len(numbers)==2:
        out = float(numbers[0]) +1j*float(numbers[1])
    elif len(numbers)==1:
        out = float(numbers[0])
    else:
        raise ValueError('couldnt parse z0 string')
    return out
    
