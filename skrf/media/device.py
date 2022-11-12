"""
.. currentmodule:: skrf.media.device

========================================
device (:mod:`skrf.media.device`)
========================================

Device is a generic n-port microwave device class to create common Network.

Device Class
---------------
.. autosummary::
    :toctree: generated/

    Device

Example Devices
---------------
.. autosummary::
    :toctree: generated/

    MatchedSymmetricCoupler
    Hybrid
    QuadratureHybrid
    Hybrid180
    DualCoupler

"""


import numpy as npy
from abc import ABCMeta, abstractmethod, abstractproperty
from .. import mathFunctions as mf
from ..network  import connect
from numpy import sqrt,exp


class Device:
    """
    A n-port microwave device
    
    Parameters 
    -----------
    media : skrf.media.Media
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, media):
        self.media = media 
        
    @abstractproperty
    def ntwk(self):
        """
        the network representation of a given device
        """
        return None
        
        
class MatchedSymmetricCoupler(Device):
    """
    A Matched Symmetric Coupler
    
    The resultant ntwk port assignment is as follows:
        * 0 - insertion
        * 1 - transmit
        * 2 - coupled 
        * 3 - isolated
    """
    def __init__(self, media, c=None, t=None, t_phase=0, phase_diff=0, 
                 nports=4, *args, **kw):
        Device.__init__(self, media=media, *args, **kw)
        
        if c is None and t is None:
            raise ValueError('Must pass either `c`  or `t`')
        
        if nports not in [3,4]:
            raise ValueError('nports must be 3 or 4')
        
        self.phase_diff = phase_diff
        self.t_phase = t_phase
        self.i = 0
        self.nports=nports
        
        
        if c is not None:
            self.c = c
        
        if t is not None:
            self.t = t
        
  
    @classmethod
    def from_dbdeg(cls, media, db, deg=0, *args,**kw):
        r"""
        Create a coupler in terms of couping(dB) and phase offset(deg)
        
        Parameters 
        -----------
        media : skrf.media.Media Object
        
        db : number or array-like
            the magnitude of the coupling value (in dB), sign dont matter
        deg : number or array-like
            phase offset between the transmit and coupled arms. defined
            as : coupled arm = transmit arm +phase offset
        \*args,\*\*kw: passed to self.__init__()    
        
        """
        c = mf.db_2_mag(-1*abs(db))
        return cls(media=media, c=c, phase_diff=deg, *args,**kw)
    
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self,c):
        self._c = c
        t_rad = mf.degree_2_radian(self.t_phase)
        c_rad = t_rad + mf.degree_2_radian(self.phase_diff)
        
        self._t = npy.sqrt(1- npy.abs(c)**2)*exp(1j*t_rad)
        self._c = c*exp(1j*c_rad)
    
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self,t):
        t_rad = mf.degree_2_radian(self.t_phase)
        c_rad = t_rad + mf.degree_2_radian(self.phase_diff)
        self._t= t*exp(1j*t_rad)
        self._c = npy.sqrt(1- npy.abs(t)**2)*exp(1j*c_rad)
    
    
    @property
    def ntwk(self):
        a = self.media.match(nports=4)
        a.s[:,0,1] = a.s[:,1,0] = a.s[:,3,2] = a.s[:,2,3] = self.t
        a.s[:,0,2] = a.s[:,2,0] = a.s[:,3,1] = a.s[:,1,3] = self.c
        a.s[:,0,3] = a.s[:,3,0] = a.s[:,1,2] = a.s[:,2,1] = self.i
        if self.nports ==3:
            match = self.media.match()
            a = connect(a,3,match,0)
        return a

    
class Hybrid(MatchedSymmetricCoupler):
    """
    A 3dB Coupler of given phase difference
    """
    def __init__(self, media, t_phase=180,phase_diff=0, *args, **kw):
        c = 1/sqrt(2)
        MatchedSymmetricCoupler.__init__(self,media=media, c=c, 
                                         phase_diff=phase_diff, 
                                         t_phase=t_phase,*args,**kw)


        
class QuadratureHybrid(MatchedSymmetricCoupler):
    """
    A 3dB Coupler with 90deg phase diff between transmit and coupled arms
    """
    def __init__(self, media,t_phase=0, *args, **kw):
        c = 1/sqrt(2)
        MatchedSymmetricCoupler.__init__(self,media=media,c=c,t_phase=t_phase,
                                         phase_diff=-90, *args, **kw)

    
class Hybrid180(Device):
    """
    180degree hybrid
    
    This device can be used to combine two signals  in and out of phase,
    or as a divider, with outputs in or out of phase.
    
    The resultant ntwk port assignment is as follows:
        * 0 - sum (A+B)
        * 1 - input A
        * 2 - input B 
        * 3 - delta (A-B)
        
        
    http://www.microwaves101.com/encyclopedias/hybrid-couplers
    """
    def __init__(self, media, nports=4, *args, **kw):
        Device.__init__(self, media=media, *args, **kw)
        self.nports = nports
    @property
    def ntwk(self):
        a = self.media.match(nports=4)
        for m,n in [(0,1),(1,0),(2,0),(0,2),(3,2),(2,3)]:
            a.s[:,m,n]=-1j
        for m,n in [(3,1),(1,3)]:
            a.s[:,m,n]=1j
        
        a.s = a.s*1/sqrt(2)
        if self.nports ==3:
            match = self.media.match()
            a = connect(a,3,match,0)
        return a
        
class DualCoupler(Device):
    """
    Pair of back-to-back directional couplers
    
    Ports are as follows:
        * 0 : insertion of coupler 1
        * 1 : insertion on coupler 2
        * 2 : coupled on coupler 1
        * 3 : coupled on coupler 2
    """
    def __init__(self, media, c1=1/sqrt(2), c2=None, c1kw={},c2kw ={}):
        Device.__init__(self,media=media)
        if c2 is None:
            c2= c1
        self.c1 = MatchedSymmetricCoupler(media=media,c=c1,nports=3,**c1kw)
        self.c2 = MatchedSymmetricCoupler(media=media,c=c2,nports=3,**c2kw)
        
    @property
    def ntwk(self):
        c1ntwk = self.c1.ntwk
        c2ntwk = self.c2.ntwk
        ntwk = connect(c1ntwk,1,c2ntwk,1)
        ntwk.renumber([0,1,2,3],[0,2,1,3])
        return ntwk
