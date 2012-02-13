
#       mathFunctions.py
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
.. currentmodule:: skrf.mathFunctions
=============================================
mathFunctions (:mod:`skrf.mathFunctions`)
=============================================


Provides commonly used mathematical functions.

Complex Component Conversion
---------------------------------
.. autosummary::
        :toctree: generated/

        complex_2_reim
        complex_2_magnitude
        complex_2_db
        complex_2_radian
        complex_2_degree
        complex_2_magnitude

Phase Unwrapping
--------------------------------
.. autosummary::
        :toctree: generated/

        unwrap_rad
        sqrt_phase_unwrap



Unit Conversion
--------------------------------
.. autosummary::
        :toctree: generated/

        radian_2_degree
        degree_2_radian
        np_2_db
        db_2_np

Scalar-Complex Conversion
---------------------------------
These conversions are useful for wrapping other functions that dont
support complex numbers.

.. autosummary::
        :toctree: generated/

        complex2Scalar
        scalar2Complex


Special Functions
---------------------------------
.. autosummary::
        :toctree: generated/

        dirac_delta
        neuman
        null

'''
import numpy as npy
from numpy import pi,angle
from scipy.fftpack import ifft, ifftshift, fftshift
from scipy import signal
## simple conversions
def complex_2_magnitude(input):
    '''
    returns the magnitude of a complex number.
    '''
    return abs(input)

def complex_2_db(input):
    '''
    returns the magnitude in dB of a complex number.

    returns:
            20*log10(|z|)
    where z is a complex number
    '''
    return magnitude_2_db(npy.abs( input))

def complex_2_radian(input):
    '''
    returns the angle complex number in radians.

    '''
    return npy.angle(input)

def complex_2_degree(input):
    '''
    returns the angle complex number in radians.

    '''
    return npy.angle(input, deg=True)

def complex_2_quadrature(z):
    '''
    takes a complex number and returns quadrature, which is (length, arc-length from real axis)
    '''
    return ( npy.abs(z), npy.angle(z)*npy.abs(z))

def complex_components(z):
    '''
    break up a complex array into all possible scalar components

    takes: complex ndarray
    return:
            c_real: real part
            c_imag: imaginary part
            c_angle: angle in degrees
            c_mag:  magnitude
            c_arc:  arclength from real axis, angle*magnitude
    '''
    return (npy.real(z), npy.imag(z), npy.angle(z,deg=True), complex_2_quadrature(z)[0], complex_2_quadrature(z)[1])

def complex_2_reim(z):
    '''
    takes:
             input: complex number or array
    return:
            real: real part of input
            imag: imaginary part of input


    note: this just calls 'complex_components'
    '''
    out = complex_components(z)
    return (out[0],out[1])

def magnitude_2_db(input):
    '''
    converts magnitude to db

     db is given by
            20*log10(|z|)
    where z is a complex number
    '''
    return  20*npy.log10(input)

def db_2_magnitude(input):
    '''
    converts db to normal magnitude

    returns:
            10**((z)/20.)
    where z is a complex number
    '''
    return 10**((input)/20.)

def db_2_np(x):
    '''
    converts a value in nepers to dB
    '''
    return (log(10)/20) * x
def np_2_db(x):
    '''
    converts a value in dB to neper's
    '''
    return 20/log(10) * x

def radian_2_degree(rad):
    return (rad)*180/pi

def degree_2_radian(deg):
    return (deg)*pi/180.

def unwrap_rad(input):
    '''
    unwraps a phase given in radians

    the normal numpy unwrap is not what you usually want for some reason
    '''
    return .5*npy.unwrap(2*input,axis=0)

def sqrt_phase_unwrap(input):
    '''
    takes the square root of a complex number with unwraped phase

    this idea came from Lihan Chen
    '''
    return npy.sqrt(abs(input))*\
            npy.exp(0.5*1j*unwrap_rad(complex_2_radian(input)))

# mathematical functions
def dirac_delta(x):
    '''
    the dirac function.

    can take numpy arrays or numbers
    returns 1 or 0 '''
    return (x==0)*1.+(x!=0)*0.
def neuman(x):
    '''
    neumans number

    2-dirac_delta(x)

    '''
    return 2. - dirac_delta(x)
def null(A, eps=1e-15):
    '''
     calculates the null space of matrix A.
    i found this on stack overflow.
     '''
    u, s, vh = npy.linalg.svd(A)
    null_space = npy.compress(s <= eps, vh, axis=0)
    return null_space.T

# old functions just for reference
def complex2Scalar(input):
    input= npy.array(input)
    output = []
    for k in input:
        output.append(npy.real(k))
        output.append(npy.imag(k))
    return npy.array(output).flatten()

def scalar2Complex(input):
    input= npy.array(input)
    output = []

    for k in range(0,len(input),2):
        output.append(input[k] + 1j*input[k+1])
    return npy.array(output).flatten()

def complex2dB(complx):
    dB = 20 * npy.log10(npy.abs( (npy.real(complx) + 1j*npy.imag(complx) )))
    return dB

def complex2ReIm(complx):
    return npy.real(complx), npy.imag(complx)

def complex2MagPhase(complx,deg=False):
    return npy.abs(complx), npy.angle(complx,deg=deg)




def psd2TimeDomain(f,y, windowType='hamming'):
    '''convert a one sided complex spectrum into a real time-signal.
    takes
            f: frequency array,
            y: complex PSD arary
            windowType: windowing function, defaults to rect

    returns in the form:
            [timeVector, signalVector]
    timeVector is in inverse units of the input variable f,
    if spectrum is not baseband then, timeSignal is modulated by
            exp(t*2*pi*f[0])
    so keep in mind units, also due to this f must be increasing left to right'''


    # apply window function
    #TODO: make sure windowType exists in scipy.signal
    if (windowType != 'rect' ):
        exec "window = signal.%s(%i)" % (windowType,len(f))
        y = y * window

    #create other half of spectrum
    spectrum = (npy.hstack([npy.real(y[:0:-1]),npy.real(y)])) + \
            1j*(npy.hstack([-npy.imag(y[:0:-1]),npy.imag(y)]))

    # do the transform
    df = abs(f[1]-f[0])
    T = 1./df
    timeVector = npy.linspace(-T/2.,T/2,2*len(f)-1)
    signalVector = ifftshift(ifft(ifftshift(spectrum)))

    #the imaginary part of this signal should be from fft errors only,
    signalVector= npy.real(signalVector)
    # the response of frequency shifting is
    # exp(1j*2*pi*timeVector*f[0])
    # but i would have to manually undo this for the inverse, which is just
    # another  variable to require. the reason you need this is because
    # you canttransform to a bandpass signal, only a lowpass.
    #
    return timeVector, signalVector
