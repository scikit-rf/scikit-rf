

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
from numpy import pi,angle,unwrap   
from scipy.fftpack import ifft, ifftshift, fftshift
from scipy import signal

global LOG_OF_NEG
LOG_OF_NEG = -100

global INF
INF = 1e99

global ALMOST_ZERO
ALMOST_ZERO = 1e-12

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

def complex_2_db10(input):
    '''
    returns the magnitude in dB of a complex number.

    returns:
            10*log10(|z|)
    where z is a complex number
    '''
    return mag_2_db10(npy.abs( input))


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

def magnitude_2_db(input,zero_nan=True):
    '''
    converts linear magnitude to db

     db is given by
            20*log10(|z|)
    where z is a complex number
    '''
    if zero_nan:
        out = 20 * npy.log10(input)
        try:
            out[npy.isnan(out)] = LOG_OF_NEG
        except (TypeError):
            # input is a number not array-like
            if npy.isnan(out):
                return LOG_OF_NEG
            
    else:
        out = 20*npy.log10(input)

    return out

mag_2_db = magnitude_2_db

def mag_2_db10(input,zero_nan=True):
    '''
    converts linear magnitude to db

     db is given by
            10*log10(|z|)
    where z is a complex number
    '''
    out = 10 * npy.log10(input)
    if zero_nan:
        try:
            out[npy.isnan(out)] = LOG_OF_NEG
        except (TypeError):
            # input is a number not array-like
            if npy.isnan(out):
                return LOG_OF_NEG
            
    
    return out

def db_2_magnitude(input):
    '''
    converts db to linear magnitude. 

    returns:
            10**((z)/20.)
    where z is a complex number
    '''
    return 10**((input)/20.)

db_2_mag = db_2_magnitude


def db10_2_mag(input):
    '''
    converts db to linear magnitude

    returns:
            10**((z)/10.)
    where z is a complex number
    '''
    return 10**((input)/10.)


def magdeg_2_reim(mag,deg):
    '''
    converts linear magnitude and phase (in deg) arrays into a complex array
    '''
    return mag*npy.exp(1j*deg*pi/180.)
    
def dbdeg_2_reim(db,deg):
    '''
    converts db magnitude and phase (in deg) arrays into a complex array
    '''
    return magdeg_2_reim(db_2_magnitude(db),deg)
    
    
def db_2_np(x):
    '''
    converts a value in nepers to dB
    '''
    return (npy.log(10)/20) * x
def np_2_db(x):
    '''
    converts a value in dB to neper's
    '''
    return 20/npy.log(10) * x

def radian_2_degree(rad):
    return (rad)*180/pi

def degree_2_radian(deg):
    return (deg)*pi/180.

def unwrap_rad(input):
    '''
    unwraps a phase given in radians

    '''
    return unwrap(input,axis=0)

def sqrt_known_sign(z_squared, z_approx):
    '''
    Returns sqrt of complex number, with sign chosen to match `z_approx`
    
    Parameters 
    -------------
    z_squared : number, array-like  
        the complex to be square-rooted
    z_approx : number, array-like
        the approximate value of z. sign of z is chosen to match that of 
        z_approx
    
    Returns
    ----------
    z : number, array-like (same type as z_squared)
        square root of z_squared. 
        
    
        
    '''
    z = npy.sqrt(z_squared)
    return npy.where(
        npy.sign(npy.angle(z)) == npy.sign(npy.angle(z_approx)), 
        z, z.conj())
    
def find_correct_sign(z1,z2,z_approx):
    '''
    Create new vector from z1, z2 choosing elements with sign matching z_approx
    
    This is used when you have to make a root choice on a complex number.
    and you know the approximate value of the root. 
    
    .. math:: 
        
        z1,z2 = \\pm \\sqrt(z^2)
        

    Parameters
    ------------
    z1 : array-like
        root 1
    z2 : array-like
        root 2
    z_approx : array-like
        approximate answer of z
    
    Returns 
    ----------
    z3 : npy.array
        array build from z1 and z2 by 
        z1 where sign(z1) == sign(z_approx), z2 else
    
    '''
    return npy.where(
    npy.sign(npy.angle(z1)) == npy.sign(npy.angle(z_approx)),z1, z2)    

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


def inf_to_num(x):
    '''
    converts inf and -inf's to large numbers

    Parameters
    ------------
    x : array-like or number
        the input array or number
    Returns
    -------
    '''
    #TODO: make this valid for complex arrays
    try:
        x[npy.isposinf(x)] = INF
        x[npy.isneginf(x)] = -1*INF

    except(TypeError):
        x = npy.array(x)
        x[npy.isposinf(x)] = INF
        x[npy.isneginf(x)] = -1*INF
    


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

def rand_c(*args):
    '''
    Creates a complex random array of shape s.
    
    The bounds on real and imaginary values are (-1,1)
    
    
    Parameters
    -----------
    s : list-like
        shape of array 
    
    Examples
    ---------
    >>> x = rf.rand_c(2,2)
    '''
    s = npy.array(args)
    return 1-2*npy.random.rand(npy.product(s)).reshape(s) + \
        1j-2j*npy.random.rand(npy.product(s)).reshape(s)


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
