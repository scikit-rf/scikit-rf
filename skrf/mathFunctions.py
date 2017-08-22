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
These conversions are useful for wrapping other functions that don't
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
from numpy import pi,angle,unwrap, real, imag, array
from scipy.fftpack import ifft, ifftshift, fftshift
from scipy import signal
from scipy.interpolate import interp1d

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
    converts a value in Nepers to dB
    '''
    return (npy.log(10)/20) * x
def np_2_db(x):
    '''
    converts a value in dB to Nepers
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
        array built from z1 and z2 by
        z1 where sign(z1) == sign(z_approx), z2 else

    '''
    return npy.where(
    npy.sign(npy.angle(z1)) == npy.sign(npy.angle(z_approx)),z1, z2)

def find_closest(z1,z2,z_approx):
    '''
    Returns z1 or z2  depending on which is  closer to z_approx


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
        array built from z1 and z2

    '''
    z1_dist = abs(z1-z_approx)
    z2_dist = abs(z2-z_approx)

    return npy.where(z1_dist<z2_dist,z1, z2)

def sqrt_phase_unwrap(input):
    '''
    takes the square root of a complex number with unwrapped phase

    this idea came from Lihan Chen
    '''
    return npy.sqrt(abs(input))*\
            npy.exp(0.5*1j*unwrap_rad(complex_2_radian(input)))


# mathematical functions
def dirac_delta(x):
    '''
    the Dirac function.

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


def cross_ratio(a,b,c,d):
    '''
    The cross ratio


    defined as

    .. math::

        \frac{(a-b)(c-d)}{(a-d)*(c-b)}


    Parameters
    -------------
    a,b,c,d : complex numbers, or arrays
        mm


    '''
    return ((a-b)*(c-d))/((a-d)*(c-b))




    

def complexify(f, name=None):
    '''
    make  f(scalar)  into f(complex)
    
    if the real/imag arguments are not first, then you may specify the
    name given to them as kwargs.
    '''
    
        
    def f_c(z, *args, **kw):
        if name is not None:
            kw_re= {name:real(z)}
            kw_im= {name:imag(z)}
            kw_re.update(kw)
            kw_im.update(kw)
            return f(*args, **kw_re)+ 1j*f(*args, **kw_im)
        else:
            return f(real(z), *args,**kw)+ 1j*f(imag(z), *args, **kw)
    return f_c



# old functions just for reference
def complex2Scalar(input):
    '''
    Serializes a list/arary of complex numbers
    
    
    produces the following output for input list `x`
    
    x[0].real, x[0].imag, x[1].real, x[1].imag, etc
    '''
    input= npy.array(input)
    output = []
    for k in input:
        output.append(npy.real(k))
        output.append(npy.imag(k))
    return npy.array(output).flatten()

def scalar2Complex(input):
    '''
    inverse of `complex2Scalar`
    '''
    input= npy.array(input)
    output = []

    for k in range(0,len(input),2):
        output.append(input[k] + 1j*input[k+1])
    return npy.array(output).flatten()

def complex2dB(complx):
    dB = 20 * npy.log10(npy.abs( (npy.real(complx) + 1j*npy.imag(complx) )))
    return dB

def flatten_c_mat(s, order ='F'):
    '''
    take a 2D (mxn) complex matrix and serialize and flatten it
    
    by default (using order='F') this generates the following 
    from a  2x2 
    
    [s11,s12;s21,s22]->[s11re,s11im,s21re,s12im, ...]
    
    Parameters
    ------------
    s : ndarray
        input 2D array 
    order : ['F','C']
        order of flattening
    '''
    return complex2Scalar(s.flatten(order='F'))


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
            y: complex PSD array
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
        exec("window = signal.%s(%i)" % (windowType,len(f)))
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
    # but I would have to manually undo this for the inverse, which is just
    # another variable to require. The reason you need this is because
    # you can't transform to a bandpass signal, only a lowpass.
    #
    return timeVector, signalVector

def rational_interp(x, y, d=4, epsilon=1e-9, axis=0):
    """
    Interpolates function using rational polynomials of degree `d`.

    Interpolating function is singular when xi is exactly one of the
    original x points. If xi is closer than epsilon to one of the original points,
    then the value at that points is returned instead.

    Implementation is based on [0]_.

    References
    ------------
    .. [0] M. S. Floater and K. Hormann, "Barycentric rational interpolation with no poles and high rates of approximation," Numer. Math., vol. 107, no. 2, pp. 315-331, Aug. 2007
    """
    n = len(x)
    w = npy.zeros(n)
    for k in range(n):
        for i in range(max(0,k-d), min(k+1, n-d)):
            p = 1.0
            for j in range(i,min(n,i+d+1)):
                if j == k:
                    continue
                p *= 1/(x[k] - x[j])
            w[k] += ((-1)**i)*p

    if axis != 0:
        raise NotImplementedError("Axis other than 0 is not implemented")

    def fx(xi):
        def find_nearest(a, values, epsilon):
            idx = npy.abs(npy.subtract.outer(a, values)).argmin(0)
            return npy.abs(a[idx] - values) < epsilon

        def find_nearest_value(a, values, y):
            idx = npy.abs(npy.subtract.outer(a, values)).argmin(0)
            return y[idx]

        nearest = find_nearest(x, xi, epsilon)
        nearest_value = find_nearest_value(x, xi, y)

        #There needs to be a cleaner way
        w_shape = [1]*len(y.shape)
        w_shape[0] = -1
        wr = w.reshape(*w_shape)

        with npy.errstate(divide='ignore', invalid='ignore'):
            #nans will be fixed later
            v = npy.sum([y[i]*wr[i]/((xi - x[i]).reshape(*w_shape)) for i in range(n)], axis=0)\
                /npy.sum([w[i]/((xi - x[i]).reshape(*w_shape)) for i in range(n)], axis=0)

        for e, i in enumerate(nearest):
            if i:
                v[e] = nearest_value[e]

        return v

    return fx
