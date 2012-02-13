
#       freeSpace.py
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
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
.. module:: skrf.media.cpw
========================================
cpw (:mod:`skrf.media.cpw`)
========================================

contains CPW class
'''
from scipy.constants import  epsilon_0, mu_0
from scipy.special import ellipk
from numpy import real, imag,pi,sqrt,log,zeros
from .media import Media
from ..tlineFunctions import skin_depth, surface_resistivity

class CPW(Media):
    '''
    Coplanar waveguide class


    This class was made from the technical documentation [#]_ provided
    by the qucs project [#]_ .
    The variables  and properties of this class are coincident with
    their derivations.

    .. [#] http://qucs.sourceforge.net/docs/technical.pdf
    .. [#] http://www.qucs.sourceforge.net/
    '''
    def __init__(self, frequency, w , s, ep_r, t=None, rho=None, \
            *args, **kwargs):
        '''
        Coplanar Waveguide  initializer

        Parameters
        -------------
        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        w : number, or array-like
                width of center conductor, in m.
        s : number, or array-like
                width of gap, in m.
        ep_r : number, or array-like
                relative permativity of substrate
        t : number, or array-like, optional
                conductor thickness, in m.
        rho: number, or array-like, optional
                resistivity of conductor (None)


        '''
        self.frequency, self.w, self.s, self.ep_r, self.t, self.rho =\
                frequency, w, s, ep_r, t, rho

        Media.__init__(self,\
                frequency = frequency,\
                propagation_constant = self.gamma, \
                characteristic_impedance = self.Z0,\
                *args, **kwargs)

    def __str__(self):
        f=self.frequency
        output =  \
                'Coplanar Waveguide Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints) + \
                '\n W= %.2em, S= %.2em'% \
                (self.w,self.s)
        return output

    def __repr__(self):
        return self.__str__()

    @property
    def ep_re(self):
        '''
        intermediary parameter. see qucs docs on cpw lines.
        '''
        return (self.ep_r+1)/2.

    @property
    def k1(self):
        '''
        intermediary parameter. see qucs docs on cpw lines.
        '''
        return self.w/(self.w +2*self.s)

    @property
    def K_ratio(self):
        '''
        intermediary parameter. see qucs docs on cpw lines.
        '''
        k1 = self.k1

        if (0 <= k1 <= 1/sqrt(2)):
            return pi/(log(2*(1+sqrt(k1))/(1-sqrt(k1)) ))
        elif (1/sqrt(2) < k1 <= 1):
            return (log(2*(1+sqrt(k1))/(1-sqrt(k1)) ))/pi




    @property
    def alpha_conductor(self):
        '''
        Losses due to conductor resistivity

        Returns
        --------
        alpha_conductor : array-like
                lossyness due to conductor losses
        See Also
        ----------
        surface_resistivity : calculates surface resistivity
        '''
        if self.rho is None or self.t is None:
            raise(AttributeError('must provide values conductivity and conductor thickness to calculate this. see initializer help'))

        t, k1, ep_re = self.t, self.k1,self.ep_re
        r_s = surface_resistivity(f=self.frequency.f, rho=self.rho, \
                mu_r=1)
        a = self.w/2.
        b = self.s+self.w/2.
        K = ellipk      # complete elliptical integral of first kind
        K_p = lambda x: ellipk(sqrt(1-x**2)) # ellipk's compliment

        return ((r_s * sqrt(ep_re)/(480*pi*K(k1)*K_p(k1)*(1-k1**2) ))*\
                (1./a * (pi+log((8*pi*a*(1-k1))/(t*(1+k1)))) +\
                 1./b * (pi+log((8*pi*b*(1-k1))/(t*(1+k1))))))




    def Z0(self):
        '''
        Characterisitc impedance
        '''
        return 30.*pi / sqrt(self.ep_re) * self.K_ratio


    def gamma(self):
        '''
        Propagation constant


        See Also
        --------
        alpha_conductor : calculates losses to conductors
        '''
        beta = 1j*2*pi*self.frequency.f*sqrt(self.ep_re*epsilon_0*mu_0)
        alpha = zeros(len(beta))
        if self.rho is not None and self.t is not None:
            alpha = self.alpha_conductor

        return beta+alpha
