"""
mathFunctions (:mod:`skrf.mathFunctions`)
=============================================


Provides commonly used mathematical functions.

Mathematical Constants
----------------------
Some convenient constants are defined in the :mod:`skrf.constants` module.

Complex Component Conversion
---------------------------------
.. autosummary::
        :toctree: generated/

        complex_2_reim
        complex_2_magnitude
        complex_2_db
        complex_2_db10
        complex_2_radian
        complex_2_degree
        complex_2_magnitude
        complex_2_quadrature

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
        cross_ratio

Various Utility Functions
--------------------------
.. autosummary::
        :toctree: generated/

        psd2TimeDomain
        rational_interp
        ifft
        irfft
        is_square
        is_symmetric
        is_Hermitian
        is_positive_definite
        is_positive_semidefinite
        get_Hermitian_transpose


"""
from typing import Callable
import numpy as npy
from numpy import pi, angle, unwrap, real, imag, array
from scipy import signal
from scipy.interpolate import interp1d

from . constants import NumberLike, INF, ALMOST_ZERO, LOG_OF_NEG

# simple conversions
def complex_2_magnitude(z: NumberLike):
    """
    Return the magnitude of the complex argument.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag : ndarray or scalar

    """
    return npy.abs(z)


def complex_2_db(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`20\log_{10}(|z|)`)..

    The magnitude in dB is defined as :math:`20\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag20dB : ndarray or scalar
    """
    return magnitude_2_db(npy.abs(z))


def complex_2_db10(z: NumberLike):
    r"""
    Return the magnitude in dB of a complex number (as :math:`10\log_{10}(|z|)`).

    The magnitude in dB is defined as :math:`10\log_{10}(|z|)`
    where :math:`z` is a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag10dB : ndarray or scalar
    """
    return mag_2_db10(npy.abs(z))


def complex_2_radian(z: NumberLike):
    """
    Return the angle complex argument in radian.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    ang_rad : ndarray or scalar
        The counterclockwise angle from the positive real axis on the complex
        plane in the range ``(-pi, pi]``, with dtype as numpy.float64.
    """
    return npy.angle(z)


def complex_2_degree(z: NumberLike):
    """
    Returns the angle complex argument in degree.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    ang_deg : ndarray or scalar
    """
    return npy.angle(z, deg=True)


def complex_2_quadrature(z: NumberLike):
    r"""
    Take a complex number and returns quadrature, which is (length, arc-length from real axis)

    Arc-length is calculated as :math:`|z| \arg(z)`.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    mag : array like or scalar
        magnitude (length)
    arc_length : array like or scalar
        arc-length from real axis: angle*magnitude
    """
    return (npy.abs(z), npy.angle(z)*npy.abs(z))


def complex_2_reim(z: NumberLike):
    """
    Return real and imaginary parts of a complex number.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    real : array like or scalar
        real part of input
    imag : array like or scalar
        imaginary part of input
    """
    return (npy.real(z), npy.imag(z))


def complex_components(z: NumberLike):
    """
    Break up a complex array into all possible scalar components.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    c_real : array like or scalar
        real part
    c_imag : array like or scalar
        imaginary part
    c_angle : array like or scalar
        angle in degrees
    c_mag : array like or scalar
        magnitude
    c_arc : array like or scalar
        arclength from real axis, angle*magnitude
    """
    return (*complex_2_reim(z), npy.angle(z,deg=True), *complex_2_quadrature(z))


def magnitude_2_db(z: NumberLike, zero_nan: bool = True):
    """
    Convert linear magnitude to dB.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers
    zero_nan : bool, optional
        Replace NaN with zero. The default is True.

    Returns
    -------
    z : number or array_like
       Magnitude in dB given by 20*log10(|z|)
    """
    out = 20 * npy.log10(z)
    if zero_nan:
        return npy.nan_to_num(out, copy=False, nan=LOG_OF_NEG, neginf=-npy.inf)
    return out

mag_2_db = magnitude_2_db


def mag_2_db10(z: NumberLike, zero_nan:bool = True):
    """
    Convert linear magnitude to dB.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers
    zero_nan : bool, optional
        Replace NaN with zero. The default is True.

    Returns
    -------
    z : array_like
       Magnitude in dB given by 10*log10(|z|)
    """
    out = 10 * npy.log10(z)
    if zero_nan:
        return npy.nan_to_num(out, copy=False, nan=LOG_OF_NEG, neginf=-npy.inf)
    return out


def db_2_magnitude(z: NumberLike):
    """
    Convert dB to linear magnitude.

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : number or array_like
        10**((z)/20) where z is a complex number
    """
    return 10**((z)/20.)

db_2_mag = db_2_magnitude


def db10_2_mag(z: NumberLike):
    """
    Convert dB to linear magnitude.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : array_like
        10**((z)/10) where z is a complex number
    """
    return 10**((z)/10.)


def magdeg_2_reim(mag: NumberLike, deg: NumberLike):
    """
    Convert linear magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    mag : number or array_like
        A complex number or sequence of real numbers
    deg : number or array_like
        A complex number or sequence of real numbers

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers

    """
    return mag*npy.exp(1j*deg*pi/180.)

def dbdeg_2_reim(db: NumberLike, deg: NumberLike):
    """
    Converts dB magnitude and phase (in deg) arrays into a complex array.

    Parameters
    ----------
    db : number or array_like
        A realnumber or sequence of real numbers
    deg : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    z : array_like
        A complex number or sequence of complex numbers
    """
    return magdeg_2_reim(db_2_magnitude(db), deg)


def db_2_np(db: NumberLike):
    """
    Converts a value in decibel (dB) to neper (Np).

    Parameters
    ----------
    db : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    np : number or array_like
        A real number of sequence of real numbers
    """
    return (npy.log(10)/20) * db


def np_2_db(x: NumberLike):
    """
    Converts a value in Nepers (Np) to decibel (dB).

    Parameters
    ----------
    np : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    db : number or array_like
        A real number of sequence of real numbers
    """
    return 20/npy.log(10) * x


def radian_2_degree(rad: NumberLike):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    rad : number or array_like
        Angle in radian

    Returns
    -------
    deg : number or array_like
        Angle in degree
    """
    return (rad)*180/pi


def degree_2_radian(deg: NumberLike):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    deg : number or array_like
        Angle in radian

    Returns
    -------
    rad : number or array_like
        Angle in degree
    """
    return (deg)*pi/180.


def feet_2_meter(feet: NumberLike = 1):
    """
    Convert length in feet to meter.

    1 foot is equal to 0.3048 meters.

    Parameters
    ----------
    feet : number or array-like, optional
        length in feet. Default is 1.

    Returns
    -------
    meter: number or array-like
        length in meter

    See Also
    --------
    meter_2_feet
    """
    return 0.3048*feet

def meter_2_feet(meter: NumberLike = 1):
    """
    Convert length in meter to feet.

    1 meter is equal to 0.3.28084 feet.

    Parameters
    ----------
    meter : number or array-like, optional
        length in meter. Default is 1.

    Returns
    -------
    feet : number or array-like
        length in feet

    See Also
    --------
    feet_2_meter
    """
    return 3.28084*meter


def db_per_100feet_2_db_per_100meter(db_per_100feet: NumberLike = 1):
    """
    Convert attenuation values given in dB/100ft to dB/100m.

    db_per_100meter = db_per_100feet * rf.meter_2_feet()

    Parameters
    ----------
    db_per_100feet : number or array-like, optional
        Attenuation in dB/ 100 ft. Default is 1.

    Returns
    -------
    db_per_100meter : number or array-like
        Attenuation in dB/ 100 m

    See Also
    --------
    meter_2_feet
    feet_2_meter
    np_2_db
    db_2_np
    """
    return db_per_100feet * 100 / feet_2_meter(100)


def unwrap_rad(phi: NumberLike):
    """
    Unwraps a phase given in radians.

    Parameters
    ----------
    phi : number or array_like
        phase in radians

    Returns
    -------
    phi : number of array_like
        unwrapped phase in radians
    """
    return unwrap(phi, axis=0)


def sqrt_known_sign(z_squared: NumberLike, z_approx: NumberLike):
    """
    Return the square root of a complex number, with sign chosen to match `z_approx`.

    Parameters
    ----------
    z_squared : number or array-like
        the complex to be square-rooted
    z_approx : number or array-like
        the approximate value of z. sign of z is chosen to match that of
        z_approx

    Returns
    -------
    z : number, array-like (same type as z_squared)
        square root of z_squared.
    """
    z = npy.sqrt(z_squared)
    return npy.where(
        npy.sign(npy.angle(z)) == npy.sign(npy.angle(z_approx)),
        z, z.conj())


def find_correct_sign(z1: NumberLike, z2: NumberLike, z_approx: NumberLike):
    r"""
    Create new vector from z1, z2 choosing elements with sign matching z_approx.

    This is used when you have to make a root choice on a complex number.
    and you know the approximate value of the root.

    .. math::

        z1,z2 = \pm \sqrt(z^2)


    Parameters
    ----------
    z1 : array-like
        root 1
    z2 : array-like
        root 2
    z_approx : array-like
        approximate answer of z

    Returns
    -------
    z3 : npy.array
        array built from z1 and z2 by
        z1 where sign(z1) == sign(z_approx), z2 else

    """
    return npy.where(
    npy.sign(npy.angle(z1)) == npy.sign(npy.angle(z_approx)),z1, z2)


def find_closest(z1: NumberLike, z2: NumberLike, z_approx: NumberLike):
    """
    Return z1 or z2  depending on which is  closer to z_approx.

    Parameters
    ----------
    z1 : array-like
        root 1
    z2 : array-like
        root 2
    z_approx : array-like
        approximate answer of z

    Returns
    -------
    z3 : npy.array
        array built from z1 and z2

    """
    z1_dist = abs(z1-z_approx)
    z2_dist = abs(z2-z_approx)

    return npy.where(z1_dist<z2_dist,z1, z2)

def sqrt_phase_unwrap(z: NumberLike):
    r"""
    Take the square root of a complex number with unwrapped phase.

    This idea came from Lihan Chen.

    .. math::

        \sqrt{|z|} \exp( \arg_{unwrap}(z) / 2 )


    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    z : number of array_like
        A complex number or sequence of complex numbers
    """
    return npy.sqrt(abs(z))*\
            npy.exp(0.5*1j*unwrap_rad(complex_2_radian(z)))


# mathematical functions
def dirac_delta(x: NumberLike):
    r"""
    Calculate Dirac function.

    Dirac function :math:`\delta(x)` defined as :math:`\delta(x)=1` if x=0,
    0 otherwise.

    Parameters
    ----------
    x : number of array_like
        A real number or sequence of real numbers

    Returns
    -------
    delta : number of array_like
        1 or 0

    References
    ----------
    https://en.wikipedia.org/wiki/Dirac_delta_function

    """
    return (x==0)*1. + (x!=0)*0.


def neuman(x: NumberLike):
    r"""
    Calculate Neumans number.

    It is defined as:

    .. math::

        2 - \delta(x)

    where :math:`\delta` is the Dirac function.

    Parameters
    ----------
    x : number or array_like
        A real number or sequence of real numbers

    Returns
    -------
    y : number or array_like
        A real number or sequence of real numbers

    See Also
    --------
    dirac_delta
    """
    return 2. - dirac_delta(x)


def null(A: npy.ndarray, eps: float = 1e-15):
    """
    Calculate the null space of matrix A.

    Parameters
    ----------
    A : array_like
    eps : float

    Returns
    -------
    null_space : array_like

    References
    ----------
    https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    """
    u, s, vh = npy.linalg.svd(A)
    null_space = npy.compress(s <= eps, vh, axis=0)
    return null_space.T


def inf_to_num(x: NumberLike):
    """
    Convert inf and -inf's to large numbers.

    Parameters
    ------------
    x : array-like or number
        the input array or number

    Returns
    -------
    x : Number of array_like
        Input without with +/- inf replaced by large numbers
    """
    x = npy.nan_to_num(x, copy=False, nan=npy.nan, posinf=INF, neginf=-1*INF)
    return x


def cross_ratio(a: NumberLike, b: NumberLike, c: NumberLike, d:NumberLike):
    r"""
    Calculate the cross ratio of a quadruple of distinct points on the real line.


    The cross ratio is defined as:


    .. math::

        r = \frac{ (a-b)(c-d) }{ (a-d)(c-b) }



    Parameters
    ----------
    a,b,c,d :  array-like or number

    Returns
    -------
    r : array-like or number

    References
    ----------
    https://en.wikipedia.org/wiki/Cross-ratio

    """
    return ((a-b)*(c-d))/((a-d)*(c-b))


def complexify(f: Callable, name: str = None):
    """
    Make a function f(scalar) into f(complex).

    If `f(x)` then it returns `f_c(z) = f(real(z)) + 1j*f(imag(z))`

    If the real/imag arguments are not first, then you may specify the
    name given to them as kwargs.

    Parameters
    ----------
    f : Callable
        function of real variable
    name : string, optional
        name of the real/imag argument names if they are not first

    Returns
    -------
    f_c : Callable
        function of a complex variable

    Examples
    --------
    >>> def f(x): return x
    >>> f_c = rf.complexify(f)
    >>> z = 0.2 -1j*0.3
    >>> f_c(z)

    """
    def f_c(z, *args, **kw):
        if name is not None:
            kw_re = {name: real(z)}
            kw_im = {name: imag(z)}
            kw_re.update(kw)
            kw_im.update(kw)
            return f(*args, **kw_re) + 1j*f(*args, **kw_im)
        else:
            return f(real(z), *args,**kw) + 1j*f(imag(z), *args, **kw)
    return f_c



# old functions just for reference
def complex2Scalar(z: NumberLike):
    """
    Serialize a list/array of complex numbers

    Parameters
    ----------
    z : number or array_like
        A complex number or sequence of complex numbers

    Returns
    -------
    re_im : array_like
        produce the following array for an input z:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    See Also
    --------
    scalar2Complex
    """
    z = npy.array(z)
    re_im = []
    for k in z:
        re_im.append(npy.real(k))
        re_im.append(npy.imag(k))
    return npy.array(re_im).flatten()

def scalar2Complex(s: NumberLike):
    """
    Unserialize a list/array of real and imag numbers into a complex array.

    Inverse of :func:`complex2Scalar`.

    Parameters
    ----------
    s : array_like
        an array with real and imaginary parts ordered as:
        z[0].real, z[0].imag, z[1].real, z[1].imag, etc.

    Returns
    -------
    z : Number or array_like
        A complex number or sequence of complex number

    See Also
    --------
    complex2Scalar
    """
    s = npy.array(s)
    z = []

    for k in range(0,len(s),2):
        z.append(s[k] + 1j*s[k+1])
    return npy.array(z).flatten()


def flatten_c_mat(s: NumberLike, order: str = 'F'):
    """
    Take a 2D (mxn) complex matrix and serialize and flatten it.

    by default (using order='F') this generates the following
    from a  2x2

    [s11,s12;s21,s22]->[s11re,s11im,s21re,s12im, ...]

    Parameters
    ------------
    s : ndarray
        input 2D array
    order : sting, optional
        either 'F' or 'C', for the order of flattening
    """
    return complex2Scalar(s.flatten(order='F'))


def rand_c(*args) -> npy.ndarray:
    """
    Creates a complex random array of shape s.

    The bounds on real and imaginary values are (-1,1)

    Parameters
    -----------
    s : list-like
        shape of array

    Examples
    ---------
    >>> x = rf.rand_c(2,2)
    """
    return 1-2*npy.random.rand(*args) + \
        1j-2j*npy.random.rand(*args)



def psd2TimeDomain(f: npy.ndarray, y: npy.ndarray, windowType: str = 'hamming'):
    """
    Convert a one sided complex spectrum into a real time-signal.

    Parameters
    ----------
    f : list or npy.ndarray
        frequency array
    y : list of npy.ndarray
        complex PSD array
    windowType: string
        windowing function, defaults to 'hamming''

    Returns
    -------
    timeVector : array_like
        inverse units of the input variable f,
    signalVector : array_like

    Note
    ----
    If spectrum is not baseband then, `timeSignal` is modulated by `exp(t*2*pi*f[0])`.
    So keep in mind units. Also, due to this, `f` must be increasing left to right.
    """


    # apply window function
    # make sure windowType exists in scipy.signal
    if callable(getattr(signal, windowType)) and (windowType != 'rect' ):
        window = getattr(signal, windowType)(len(f))
        y = y * window

    #create other half of spectrum
    spectrum = (npy.hstack([npy.real(y[:0:-1]),npy.real(y)])) + \
            1j*(npy.hstack([-npy.imag(y[:0:-1]),npy.imag(y)]))

    # do the transform
    df = abs(f[1]-f[0])
    T = 1./df
    timeVector = npy.linspace(-T/2.,T/2,2*len(f)-1)
    signalVector = npy.fft.ifftshift(npy.fft.ifft(npy.fft.ifftshift(spectrum)))

    #the imaginary part of this signal should be from fft errors only,
    signalVector= npy.real(signalVector)
    # the response of frequency shifting is
    # exp(1j*2*pi*timeVector*f[0])
    # but I would have to manually undo this for the inverse, which is just
    # another variable to require. The reason you need this is because
    # you can't transform to a bandpass signal, only a lowpass.
    #
    return timeVector, signalVector


def rational_interp(x: npy.ndarray, y: npy.ndarray, d: int = 4, epsilon: float = 1e-9, axis: int = 0, assume_sorted: bool = False) -> Callable:
    """
    Interpolates function using rational polynomials of degree `d`.

    Interpolating function is singular when xi is exactly one of the
    original x points. If xi is closer than epsilon to one of the original points,
    then the value at that points is returned instead.

    Implementation is based on [#]_.

    Parameters
    ----------
    x : npy.ndarray
    y : npy.ndarray
    d : int, optional
        order of the polynomial, by default 4
    epsilon : float, optional
        numerical tolerance, by default 1e-9
    axis : int, optional
        axis to operate on, by default 0
    assume_sorted : bool, optional
        If False, values of x can be in any order and they are sorted first.
        If True, x has to be an array of monotonically increasing values.

    Returns
    -------
    fx : Callable
        Interpolate function

    Raises
    ------
    NotImplementedError
        if axis != 0.

    References
    ------------
    .. [#] M. S. Floater and K. Hormann, "Barycentric rational interpolation with no poles and high rates of approximation," Numer. Math., vol. 107, no. 2, pp. 315-331, Aug. 2007
    """
    if axis != 0:
        raise NotImplementedError("Axis other than 0 is not implemented")

    if not assume_sorted:
        sort_indices = npy.argsort(x, axis=axis)
        x = x[sort_indices]
        y = y[sort_indices]

    n = len(x)
    if n <= d:
        raise ValueError('Not enough x-axis points')

    w = npy.zeros(n)
    # Scaling to give close to 1 weights
    hd = (x[n//2] - x[n//2-1])**d
    for k in range(n):
        for i in range(max(0,k-d), min(k+1, n-d)):
            p = hd
            for j in range(i,min(n,i+d+1)):
                if j == k:
                    continue
                p *= 1/(x[k] - x[j])
            if i % 2 == 1:
                w[k] -= p
            else:
                w[k] += p

    # Add dimensions to match y shape
    w_shape = [1]*len(y.shape)
    w_shape[0] = -1
    w = w.reshape(w_shape)

    def fx(xi):
        # The method will divide by zero if new x value is exactly existing x value.
        # To avoid this we need to check for too close values and replace them with
        # y value at that position.
        idx = npy.searchsorted(x, xi)
        idx[idx == len(x)] = len(x) - 1
        nearest_idx = npy.where(npy.abs(x[idx] - xi) < epsilon)[0]
        nearest_value = y[idx[nearest_idx]]

        xi = xi.reshape(*w_shape)
        with npy.errstate(divide='ignore', invalid='ignore'):
            assert axis == 0
            v = sum(y[i]*w[i]/(xi - x[i]) for i in range(n))\
                /sum(w[i]/(xi - x[i]) for i in range(n))

        # Fix divide by zero errors
        for e, i in enumerate(nearest_idx):
            v[i] = nearest_value[e]

        return v

    return fx

def ifft(x: npy.ndarray) -> npy.ndarray:
    """
    Transforms S-parameters to time-domain bandpass.

    Parameters
    ----------
    x : array_like
        S-parameters vs frequency array.

    Returns
    -------
    X : array_like
        Fourier transformed array

    See Also
    --------
    irfft
    """
    return npy.fft.fftshift(npy.fft.ifft(x, axis=0), axes=0)


def irfft(x: npy.ndarray, n:int = None) -> npy.ndarray:
    """
    Transforms S-parameters to time-domain, assuming complex conjugates for
    values corresponding to negative frequencies.

    Parameters
    ----------
    x : array_like
        S-parameters vs frequency array.
    n : int, optional
        n parameter passed to :func:`numpy.fft.irfft`. Defaults to None.

    Returns
    -------
    X : array_like
        Fourier transformed array

    See Also
    --------
    ifft
    """
    return npy.fft.fftshift(npy.fft.irfft(x, axis=0, n=n), axes=0)


def is_square(mat: npy.ndarray) -> bool:
    """
    Tests whether mat is a square matrix.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for being square

    Returns
    -------
    res : boolean

    See Also
    --------
    is_unitary
    is_symmetric
    """
    return mat.shape[0] == mat.shape[1]


def is_unitary(mat: npy.ndarray, tol: float = ALMOST_ZERO) -> bool:
    """
    Tests mat for unitariness.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for unitariness
    tol : float
        Absolute tolerance. Defaults to :data:`ALMOST_ZERO`

    Returns
    -------
    res : boolean or array of boolean

    See Also
    --------
    is_square
    is_symmetric

    """
    if not is_square(mat):
        return False
    return npy.allclose(get_Hermitian_transpose(mat) @ mat,
                        npy.identity(mat.shape[0]), atol=tol)


def is_symmetric(mat: npy.ndarray, tol: int = ALMOST_ZERO) -> bool:
    """
    Tests mat for symmetry.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for symmetry
    tol : float, optional
        Absolute tolerance. Defaults to :data:`ALMOST_ZERO`

    Returns
    -------
    res : boolean or array of boolean

    See Also
    --------
    is_square
    is_unitary
    """
    if not is_square(mat):
        return False
    return npy.allclose(mat, mat.transpose(), atol=tol)


def get_Hermitian_transpose(mat: npy.ndarray) -> npy.ndarray:
    """
    Returns the conjugate transpose of mat.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to compute the conjugate transpose of

    Returns
    -------
    mat : npy.ndarray

    """
    return mat.transpose().conjugate()


def is_Hermitian(mat: npy.ndarray, tol: float = ALMOST_ZERO) -> bool:
    """
    Tests whether mat is Hermitian.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for being Hermitian
    tol : float
        Absolute tolerance

    Returns
    -------
    res : boolean

    See Also
    --------
    is_positive_definite
    is_positive_semidefinite
    """
    if not is_square(mat):
        return False
    return npy.allclose(mat, get_Hermitian_transpose(mat), atol=tol)


def is_positive_definite(mat: npy.ndarray, tol: float = ALMOST_ZERO) -> bool:
    """
    Tests mat for positive definiteness.

    Verifying that
    (1) mat is symmetric
    (2) it's possible to compute the Cholesky decomposition of mat.

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for positive definiteness
    tol : float, optional
        Absolute tolerance. Defaults to :data:`ALMOST_ZERO`

    Returns
    -------
    res : bool or array of bool

    See Also
    --------
    is_Hermitian
    is_positive_semidefinite
    """
    if not is_Hermitian(mat, tol=tol):
        return False
    try:
        npy.linalg.cholesky(mat)
        return True
    except npy.linalg.LinAlgError:
        return False


def is_positive_semidefinite(mat: npy.ndarray, tol: float = ALMOST_ZERO) -> bool:
    """
    Tests mat for positive semidefiniteness.

    Checking whether all eigenvalues of mat are nonnegative within a certain tolerance

    Parameters
    ----------
    mat : npy.ndarray
        Matrix to test for positive semidefiniteness
    tol : float, optional
        Absolute tolerance in determining nonnegativity due to loss of precision
        when computing the eigenvalues of mat. Defaults to :data:`ALMOST_ZERO`

    Returns
    -------
    res : bool or array of bool

    See Also
    --------
    is_Hermitian
    is_positive_definite
    """
    if not is_Hermitian(mat):
        return False
    try:
        v = npy.linalg.eigvalsh(mat)
    except npy.linalg.LinAlgError:
        return False
    return npy.all(v > -tol)

def rsolve(A: npy.ndarray, B: npy.ndarray) -> npy.ndarray:
    r"""Solves x @ A = B.

    Calls numpy.linalg.solve with transposed matrices.

    Same as B @ npy.linalg.inv(A) but avoids calculating the inverse and
    should be numerically slightly more accurate.

    Input should have dimension of similar to (nfreqs, nports, nports).

    Parameters
    ----------
    A : npy.ndarray
    B : npy.ndarray

    Returns
    -------
    x : npy.ndarray
    """
    return npy.transpose(npy.linalg.solve(npy.transpose(A, (0, 2, 1)).conj(),
            npy.transpose(B, (0, 2, 1)).conj()), (0, 2, 1)).conj()

def nudge_eig(mat: npy.ndarray, cond: float = 1e-9, min_eig: float = 1e-12) -> npy.ndarray:
    r"""Nudge eigenvalues with absolute value smaller than
    max(cond * max(eigenvalue), min_eig) to that value.
    Can be used to avoid singularities in solving matrix equations.

    Input should have dimension of similar to (nfreqs, nports, nports).

    Parameters
    ----------
    mat : npy.ndarray
        Matrices to nudge
    cond : float, optional
        Minimum eigenvalue ratio compared to the maximum eigenvalue
    min_eig : float, optional
        Minimum eigenvalue
    Returns
    -------
    res : npy.ndarray
        Nudged matrices
    """
    # Eigenvalues and vectors
    eigw, eigv = npy.linalg.eig(mat)
    # Max eigenvalue for each frequency
    max_eig = npy.amax(npy.abs(eigw), axis=1)
    # Calculate mask for positions where problematic eigenvalues are
    mask = npy.logical_or(npy.abs(eigw) < cond * max_eig[:, None], npy.abs(eigw) < min_eig)
    if not mask.any():
        # Nothing to do. Return the original array.
        return mat

    mask_cond = cond * npy.repeat(max_eig[:, None], mat.shape[-1], axis=-1)[mask]
    mask_min = min_eig * npy.ones(mask_cond.shape)
    # Correct the eigenvalues
    eigw[mask] = npy.maximum(mask_cond, mask_min)

    # Now assemble the eigendecomposited matrices back
    e = npy.zeros_like(mat)
    npy.einsum('ijj->ij', e)[...] = eigw
    return rsolve(eigv, eigv @ e)
