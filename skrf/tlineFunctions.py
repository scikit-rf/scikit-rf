
#       tlinefunctions.py
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
.. module:: skrf.tlineFunctions
===============================================
tlineFunctions (:mod:`skrf.tlineFunctions`)
===============================================

This module provides functions related to transmission line theory.

Impedance and Reflection Coefficient
--------------------------------------
These functions relate basic tranmission line quantities such as
characteristic impedance, input impedance, reflection coefficient, etc.
Each function has two names. One is a long-winded but readable name and
the other is a short-hand variable-like names. Below is a table relating
these two names with each other as well as common mathematical symbols.

====================  ======================  ================================
Symbol                Variable Name           Long Name
====================  ======================  ================================
:math:`Z_l`           z_l                     load_impedance
:math:`Z_{in}`        z_in                    input_impedance
:math:`\Gamma_0`      Gamma_0                 reflection_coefficient
:math:`\Gamma_{in}`   Gamma_in                reflection_coefficient_at_theta
:math:`\\theta`        theta                   electrical_length
====================  ======================  ================================

There may be a bit of confusion about the difference between the load
impedance the input impedance. This is because the load impedance **is**
the input impedance at the load. An illustration may provide some
useful reference.

Below is a (bad) illustration of a section of uniform transmission line
of characteristic impedance :math:`Z_0`, and electrical length
:math:`\\theta`. The line is terminated on the right with some
load impedance, :math:`Z_l`. The input impedance :math:`Z_{in}` and
input reflection coefficient :math:`\\Gamma_{in}` are
looking in towards the load from the distance :math:`\\theta` from the
load.

.. math::
        Z_0, \\theta

        \\text{o===============o=}[Z_l]

        \\to\\qquad\\qquad\\qquad\\quad\\qquad \\qquad \\to \\qquad \\quad

        Z_{in},\\Gamma_{in}\\qquad\\qquad\\qquad\\qquad\\quad Z_l,\\Gamma_0 \\qquad

So, to clarify the confusion,

.. math::
        Z_{in}= Z_{l},\\qquad\\qquad
        \\Gamma_{in}=\\Gamma_l \\text{ at }  \\theta=0


Short names
+++++++++++++
.. autosummary::
        :toctree: generated/

        theta
        zl_2_Gamma0
        Gamma0_2_zl
        zl_2_zin
        zl_2_Gamma_in
        Gamma0_2_Gamma_in
        Gamma0_2_zin

Long-names
++++++++++++++
.. autosummary::
        :toctree: generated/

        distance_2_electrical_length
        electrical_length_2_distance

        reflection_coefficient_at_theta
        reflection_coefficient_2_input_impedance
        reflection_coefficient_2_input_impedance_at_theta

        input_impedance_at_theta
        load_impedance_2_reflection_coefficient
        load_impedance_2_reflection_coefficient_at_theta




Distributed Circuit and Wave Quantities
----------------------------------------
.. autosummary::
        :toctree: generated/

        distributed_circuit_2_propagation_impedance
        propagation_impedance_2_distributed_circuit

Transmission Line Physics
---------------------------------
.. autosummary::
        :toctree: generated/

        skin_depth
        surface_resistivity
'''

import numpy as npy
from numpy import pi, sqrt, exp, array,tan,sin,cos,inf, log, real,imag,\
         interp, linspace, shape,zeros, reshape

from scipy.constants import mu_0
import mathFunctions as mf

INF = 1e99
ONE = 1.0 + 1/1e14


def skin_depth(f,rho, mu_r):
    '''

    the skin depth for a material.

    see www.microwaves101.com for more info.

    Parameters
    ----------
    f : number or array-like
            frequency, in Hz
    rho : number of array-like
            bulk resistivity of material, in ohm*m
    mu_r : number or array-like
            relative permiability of material

    Returns
    ----------
    skin depth : number or array-like
            the skin depth, in m

    '''
    return sqrt(rho/(pi*f*mu_r*mu_0))

def surface_resistivity(f,rho,mu_r):
    '''
    surface resistivity.

    see www.microwaves101.com for more info.

    Parameters
    ----------
    f : number or array-like
            frequency, in Hz
    rho : number or array-like
            bulk resistivity of material, in ohm*m
    mu_r : number or array-like
            relative permiability of material

    Returns
    ----------
    surface resistivity: ohms/square


    '''
    return rho/skin_depth(rho=rho,f = f, mu_r=mu_r)

def distributed_circuit_2_propagation_impedance( distributed_admittance,\
        distributed_impedance):
    '''
    Converts distrubuted circuit values to wave quantities.

    This converts complex distributed impedance and admittance to
    propagation constant and characteristic impedance. The relation is

    .. math::
            Z_0 = \\sqrt{ \\frac{Z^{'}}{Y^{'}}}
            \\quad\\quad
            \\gamma = \\sqrt{ Z^{'}  Y^{'}}

    Parameters
    ------------
    distributed_admittance : number, array-like
            distributed admittance
    distributed_impedance :  number, array-like
            distributed impedance

    Returns
    ----------
    propagation_constant : number, array-like
            distributed impedance
    characteristic_impedance : number, array-like
            distributed impedance

    See Also
    ----------
            propagation_impedance_2_distributed_circuit : opposite conversion
    '''
    propagation_constant = \
            sqrt(distributed_impedance*distributed_admittance)
    characteristic_impedance = \
            sqrt(distributed_impedance/distributed_admittance)
    return (propagation_constant, characteristic_impedance)

def propagation_impedance_2_distributed_circuit(propagation_constant, \
        characteristic_impedance):
    '''
    Converts wave quantities to distrubuted circuit values.

    Converts complex propagation constant and characteristic impedance
    to distributed impedance and admittance. The relation is,

    .. math::
            Z^{'} = \\gamma  Z_0 \\quad\\quad
            Y^{'} = \\frac{\\gamma}{Z_0}

    Parameters
    ------------
    propagation_constant : number, array-like
            distributed impedance
    characteristic_impedance : number, array-like
            distributed impedance

    Returns
    ----------
    distributed_admittance : number, array-like
            distributed admittance
    distributed_impedance :  number, array-like
            distributed impedance


    See Also
    ----------
            distributed_circuit_2_propagation_impedance : opposite conversion
    '''
    distributed_admittance = propagation_constant/characteristic_impedance
    distributed_impedance = propagation_constant*characteristic_impedance
    return (distributed_admittance,distributed_impedance)

def electrical_length(gamma, f , d, deg=False):
    '''
    Calculates the electrical length of a section of transmission line.

    .. math::
            \\theta = \\gamma(f) \\cdot d

    Parameters
    ----------
    gamma : function
            propagation constant function, which takes frequency in hz as a
            sole argument. see Notes.
    l : number or array-like
            length of line, in meters
    f : number or array-like
            frequency at which to calculate
    deg : Boolean
            return in degrees or not.

    Returns
    ----------
    theta :  number or array-like
            electrical length in radians or degrees, depending on  value of
            deg.

    See Also
    -----------
            electrical_length_2_distance : opposite conversion

    Notes
    ------
    the convention has been chosen that forward propagation is
    represented by the positive imaginary part of the value returned by
    the gamma function
    '''

    # typecast to a 1D array
    f = array(f, dtype=float).reshape(-1)
    d = array(d, dtype=float).reshape(-1)

    if deg == False:
        return  gamma(f)*d
    elif deg == True:
        return  mf.radian_2_degree(gamma(f)*d )

def electrical_length_2_distance(theta, gamma, f0,deg=True):
    '''
    Convert electrical length to a physical distance.

    .. math::
            d = \\frac{\\theta}{\\gamma(f_0)}

    Parameters
    ----------
    theta : number or array-like
            electical length. units depend on `deg` option
    gamma : function
            propagation constant function, which takes frequency in hz as a
            sole argument. see Notes
    f0 : number or array-like
            frequency at which to calculate
    deg : Boolean
            return in degrees or not.

    Returns
    ----------
    d: physical distance

    Notes
    ------
    the convention has been chosen that forward propagation is
    represented by the positive imaginary part of the value returned by
    the gamma function

    See Also
    ---------
            distance_2_electrical_length: opposite conversion
    '''
    if deg == True:
        theta = mf.degree_2_radian(theta)
    return theta/imag(gamma(f0))

def load_impedance_2_reflection_coefficient(z0, zl):
    '''
    Returns the reflection coefficient for a given load impedance, and
    characteristic impedance.

    For a transmission line of characteristic impedance :math:`Z_0`
    terminated with load impedance :math:`Z_l`, the complex reflection
    coefficient is given by,

    .. math::
            \\Gamma = \\frac {Z_l - Z_0}{Z_l + Z_0}


    Parameters
    ----------
    z0 :  number or array-like
            characteristic impedance
    zl :  number or array-like
            load impedance (aka input impedance)


    Returns
    --------
    gamma : number or array-like
            reflection coefficient

    See Also
    ----------
            Gamma0_2_zl : reflection coefficient to load impedance


    Notes
    ------
            inputs are typecasted to 1D complex array
    '''
    # typecast to a complex 1D array. this makes everything easier
    z0 = array(z0, dtype=complex).reshape(-1)
    zl = array(zl, dtype=complex).reshape(-1)

    # handle singularity  by numerically representing inf as big number
    zl[(zl==npy.inf)] = INF

    return ((zl -z0 )/(zl+z0))

def reflection_coefficient_2_input_impedance(z0,Gamma):
    '''
    calculates the input impedance given a reflection coefficient and
    characterisitc impedance

    .. math::
            Z_0 (\\frac {1 + \\Gamma}{1-\\Gamma})



    Parameters
    ----------
    Gamma : number or array-like
            complex reflection coefficient
    z0 : number or array-like
            characteristic impedance

    Returns
    --------
    zin : number or array-like
            input impedance


    '''
    # typecast to a complex 1D array. this makes everything easier
    Gamma = array(Gamma, dtype=complex).reshape(-1)
    z0 = array(z0, dtype=complex).reshape(-1)

    #handle singularity by numerically representing inf as close to 1
    Gamma[(Gamma == 1)] = ONE

    return z0*((1.0+Gamma )/(1.0-Gamma))

def reflection_coefficient_at_theta(Gamma0,theta):
    '''
    reflection coefficient at a given electrical length.

    .. math::
            \\Gamma_{in} = \\Gamma_0 e^{-2j\\theta}

    Parameters
    ----------
    Gamma0 : number or array-like
            reflection coefficient at theta=0
    theta : number or array-like
            electrical length, (may be complex)

    Returns
    ----------
    Gamma_in : number or array-like
            input reflection coefficient

    '''
    Gamma0 = array(Gamma0, dtype=complex).reshape(-1)
    theta = array(theta, dtype=complex).reshape(-1)
    return Gamma0 * exp(2j* theta)

def input_impedance_at_theta(z0,zl, theta):
    '''
    input impedance of load impedance zl at a given electrical length,
    given characteristic impedance z0.

    Parameters
    ----------
    z0 : characteristic impedance.
    zl : load impedance
    theta : electrical length of the line, (may be complex)

    Returns
    ---------
    '''
    Gamma = load_impedance_2_reflection_coefficient(z0=z0,zl=zl)
    Gamma_in = reflection_coefficient_at_theta(Gamma=Gamma, theta=theta)
    return reflection_coefficient_2_input_impedance(z0=z0, Gamma=Gamma_in)

def load_impedance_2_reflection_coefficient_at_theta(z0, zl, theta):
    Gamma0 = load_impedance_2_reflection_coefficient(z0=z0,zl=zl)
    Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
    return Gamma_in

def reflection_coefficient_2_input_impedance_at_theta(z0, Gamma0, theta):
    '''
    calculates the input impedance at electrical length theta, given a
    reflection coefficient and characterisitc impedance of the medium
    Parameters
    ----------
            z0 - characteristic impedance.
            Gamma: reflection coefficient
            theta: electrical length of the line, (may be complex)
    returns
            zin: input impedance at theta
    '''
    Gamma_in = reflection_coefficient_at_theta(Gamma0=Gamma0, theta=theta)
    zin = reflection_coefficient_2_input_impedance(z0=z0,Gamma=Gamma_in)
    return zin
# short hand convinience.
# admitantly these follow no logical naming scheme, but they closely
# correspond to common symbolic conventions, and are convenient
theta = electrical_length
distance_2_electrical_length = electrical_length

zl_2_Gamma0 = load_impedance_2_reflection_coefficient
Gamma0_2_zl = reflection_coefficient_2_input_impedance

zl_2_zin = input_impedance_at_theta
zl_2_Gamma_in = load_impedance_2_reflection_coefficient_at_theta

Gamma0_2_Gamma_in = reflection_coefficient_at_theta
Gamma0_2_zin = reflection_coefficient_2_input_impedance_at_theta
