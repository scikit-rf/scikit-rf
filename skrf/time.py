"""
.. module:: skrf.time
========================================
time (:mod:`skrf.time`)
========================================

Time domain functions


.. autosummary::
   :toctree: generated/

   time_gate
   detect_span
   find_n_peaks
   indexes

"""
import scipy.signal

from .util import find_nearest_index
from scipy import signal
import numpy as npy
from numpy.fft import fft, rfft, fftshift, ifft, irfft, ifftshift
from scipy.ndimage import convolve1d
import warnings

# imports for type hinting
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from .network import Network


def indexes(y: npy.ndarray, thres: float = 0.3, min_dist: int = 1) -> npy.ndarray:
    """
    Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.], optional
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected. Default is 0.3
    min_dist : int, optional
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint. Default is 1

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected

    Notes
    -----
    This function was taken from peakutils-1.1.0
    http://pythonhosted.org/PeakUtils/index.html

    """
    #This function was taken from peakutils, and is covered
    # by the MIT license, included below:

    #The MIT License (MIT)

    #Copyright (c) 2014 Lucas Hermann Negri

    #Permission is hereby granted, free of charge, to any person obtaining a copy
    #of this software and associated documentation files (the "Software"), to deal
    #in the Software without restriction, including without limitation the rights
    #to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    #copies of the Software, and to permit persons to whom the Software is
    #furnished to do so, subject to the following conditions:

    #The above copyright notice and this permission notice shall be included in
    #all copies or substantial portions of the Software.

    #THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    #IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    #FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    #AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    #LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    #OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    #THE SOFTWARE.

    if isinstance(y, npy.ndarray) and npy.issubdtype(y.dtype, npy.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (npy.max(y) - npy.min(y)) + npy.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = npy.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = npy.where(dy == 0)

    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return npy.array([])

    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = npy.hstack([dy[1:], 0.])
        zerosl = npy.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros, = npy.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros] = zerosl[zeros]
        zeros, = npy.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = npy.where((npy.hstack([dy, 0.]) < 0.)
                     & (npy.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[npy.argsort(y[peaks])][::-1]
        rem = npy.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = npy.arange(y.size)[~rem]

    return peaks


def find_n_peaks(x: npy.ndarray, n: int, thres: float = 0.9, **kwargs) -> List[int]:
    """
    Find a given number of peaks in a signal.

    Parameters
    ----------
    x : npy.ndarray
        signal
    n : int
        number of peaks to search for
    thres : float, optional
        threshold, default is 0.9
    **kwargs : optional keyword arguments passed to :func:`indexes`

    Returns
    -------
    peak_idxs : list of int
        List containing the numeric indexes of the peaks that were detected

    Raises
    ------
    ValueError
        If no peaks are found.
    """
    for dummy in range(10):

        idx = indexes(x, **kwargs)
        if len(idx) < n:
            thres *= .5

        else:
            peak_vals = sorted(x[idx], reverse=True)[:n]
            peak_idxs = [x.tolist().index(k) for k in peak_vals]

            return peak_idxs
    raise ValueError('Couldnt find %i peaks' % n)

time_lookup_dict = {
    "s": 1,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12
}

def detect_span(ntwk: 'Network', t_unit: str = "") -> float:
    """
    Detect the correct time-span between two largest peaks.

    Parameters
    ----------
    ntwk : :class:`~skrf.network.Network`
        network to get data from
    
    t_unit : str
        Time unit for start, stop, center and span arguments, defaults to nanoseconds (ns).
        
        Possible values:
            * 's': seconds
            * 'ms': milliseconds
            * 'µs' or 'us': microseconds
            * 'ns': nanoseconds (default)
            * 'ps': picoseconds


    Returns
    -------
    span : float in unit t_unit
    """

    if t_unit == "":
        # Do not raise in autogate mode, where all parameters are None
        warnings.warn('''
                        Time unit not passed: currently uses 'ns' per default.
                        The future versions of scikit-rf will use 's' per default instead,
                        so it is recommended to specify explicitly the time unit
                        to obtain similar results with future versions.
                        ''',
                        DeprecationWarning, stacklevel=2)
        t_unit = 'ns'

    x = ntwk.s_time_db.flatten()
    p1, p2 = find_n_peaks(x, n=2)
    print(p1, p2)
    # distance to nearest neighbor peak
    span = abs(ntwk.frequency.t[p1]-ntwk.frequency.t[p2])

    return span / time_lookup_dict[t_unit]



def time_gate(ntwk: 'Network', start: float = None, stop: float = None, center: float = None, span: float = None,
              mode: str = 'bandpass', window=('kaiser', 6),
              method: str ='fft', fft_window: str='hann', conv_mode: str='wrap', t_unit: str = "") -> 'Network':
    """
    Time-domain gating of one-port s-parameters with a window function from scipy.signal.windows.

    The gate can be defined with start/stop times, or by center/span. All times are in units of nanoseconds but
    can be changed using the `t_unit` parameter. The default unit will change to `s` in **scikit-rf** version 1.0.
    Common windows are:

    * ('kaiser', 6)
    * 6 # integers are interpreted as kaiser beta-values
    * 'hamming'
    * 'boxcar'  # a straight up rect

    If no parameters are passed this will try to auto-gate the largest
    peak.

    Parameters
    ----------
    ntwk : :class:`~skrf.network.Network`
        network to operate on
    start : number, or None
        start of time gate in t_unit.
    stop : number, or None
        stop of time gate in t_unit.
    center : number, or None
        center of time gate, in t_unit. If None, and span is given,
        the gate will be centered on the peak.
    span : number, or None
        span of time gate, in t_unit.  If None span will be half of the
        distance to the second tallest peak
    mode : ['bandpass', 'bandstop']
        mode of gate
    window : string, float, or tuple
        passed to `window` arg of `scipy.signal.get_window()`
    method : str
        Gating method. There are 3 option: 'convolution', 'fft', 'rfft'.

        With *'convolution'*, the time-domain gate gets transformed into frequency-domain using inverse FFT and the
        gating is then achieved by convolution with the frequency-domain data.

        With *'fft'* (default), the data gets transformed into time-domain using inverse FFT and the gating is achieved
        by multiplication with the time-domain gate. The gated time-domain signal is then transformed back into
        frequency-domain using inverse FFT. As only positive signal frequencies are considered for the inverse FFT
        (with or without a dc component), the resulting time-domain signal has the same number of samples as in the
        frequency-domain, but is complex-valued. This method is also know as *time-domain band-pass mode*.

        With *'rfft'*, the procedure is the same as with *'fft'*, but the inverse FFT uses a complex-conjugate copy of
        the positive signal frequencies for the negative frequencies (Hermitian frequency response). A dc sample is
        also required. The resulting time-domain signal is real-valued and has twice the number of samples, which gives
        an improved time resolution. This method is also known as *time-domain low-pass mode*.

    fft_window : str or tuple or None
        Frequency-domain window applied before the inverse FFT in case of the (R)FFT method.
        This parameter takes the same values as the `window` parameter.
        Example: `window='hann` (default), or `window=('kaiser', 5)`, or `window=None`.
        The window helps to remove artefacts such as time-domain sidelobes of the pulses, but it is a trade-off with
        the achievable pulse width. The window is removed when the gated time-domain signals is transformed back into
        frequency-domain.

    conv_mode : str
        Extension mode for the convolution (if selected) determining how the frequency-domain gate is extended beyond
        the boundaries. This has a large effect on the generation of gating artefacts due to boundary effects. The
        optimal mode depends on the data. See the parameter description of `scipy.ndimage.convolve1d` for the available
        options.
    
    t_unit : str
        Time unit for start, stop, center and span arguments, defaults to nanoseconds (ns).
        
        Possible values:
            * 's': seconds
            * 'ms': milliseconds
            * 'µs' or 'us': microseconds
            * 'ns': nanoseconds (default)
            * 'ps': picoseconds


    Note
    ----
    You cant gate things that are 'behind' strong reflections. This
    is due to the multiple reflections that occur.

    If you need to time-gate an N-port network, then you should
    gate each s-parameter independently.

    Returns
    -------
    ntwk : Network
        copy of ntwk with time-gated s-parameters

    .. warning::
        Depending on sharpness of the gate, the band edges may be
        inaccurate, due to properties of FFT. We do not re-normalize
        anything.
    """

    if ntwk.nports >1:
        raise ValueError('Time-gating only works on one-ports. Try passing `ntwk.s11` or `ntwk.s21`.')

    if t_unit == "":
        if not all([e is None for e in [start, stop, center, span]]):
            # Do not raise in autogate mode, where all parameters are None
            warnings.warn('''
                            Time unit not passed: currently uses 'ns' per default.
                            The future versions of scikit-rf will use 's' per default instead,
                            so it is recommended to specify explicitly the time unit
                            to obtain similar results with future versions.
                            ''',
                            DeprecationWarning, stacklevel=2)
        t_unit = 'ns'

    t_mult = time_lookup_dict[t_unit]

    if start is not None and stop is not None:
        start *= t_mult
        stop *= t_mult
        span = abs(stop-start)
        center = (stop+start)/2.

    else:
        if center is None:
            # they didnt provide center, so find the peak
            n = ntwk.s_time_mag.argmax()
            center = ntwk.frequency.t_ns[n]

        if span is None:
            span = detect_span(ntwk, t_unit='ns')

        center *= t_mult
        span *= t_mult
        start = center - span / 2.
        stop = center + span / 2.

    ntwk_gated = ntwk.copy()
    method = method.lower()
    n_fd = ntwk.frequency.npoints
    df = ntwk.frequency.step

    if method == 'convolution':
        # frequency-domain gating
        n_td = n_fd
        # create dummy-window
        window_fd = npy.ones(n_fd)

    elif method == 'fft':
        # time-domain band-pass mode
        n_td = n_fd
        if fft_window is not None:
            # create band-pass window (zero on both lower and upper limit, one at center)
            window_fd = signal.get_window(window, n_fd)
        else:
            # create dummy-window
            window_fd = npy.ones(n_fd)

    elif method == 'rfft':
        # time-domain low-pass mode
        if ntwk.f[0] > 0.0:
            # no dc point included
            warnings.warn('The network data to be gated does not contain the dc point (0 Hz). This is required for the '
                          'selected low-pass gating mode. Please consider to include the dc point if the results are '
                          'inaccurate, either by direct measurement of by extrapolation using '
                          'skrf.Network.extrapolate_to_dc().', UserWarning, stacklevel=2)
        n_td = 2 * n_fd - 1
        if fft_window is not None:
            # create low-pass window (one at lower limit at f=0, zero on upper limit)
            window_fd = signal.get_window(window, 2 * n_fd)
            window_fd = window_fd[n_fd:]
        else:
            # create dummy-window
            window_fd = npy.ones(n_fd)

    else:
        raise ValueError('Invalid parameter method=`{}`'.format(method))

    # apply frequency-domain window
    ntwk_gated.s[:, 0, 0] = ntwk_gated.s[:, 0, 0] * window_fd

    # create time vector
    t = npy.linspace(-0.5 / df, 0.5 / df, n_td)

    # find start/stop gate indices
    start_idx = find_nearest_index(t, start)
    stop_idx = find_nearest_index(t, stop)

    # create gating window
    window_width = abs(stop_idx - start_idx)
    window = signal.get_window(window, window_width)

    # create the gate by padding the window with zeros
    gate = npy.zeros_like(t)
    gate[start_idx:stop_idx] = window

    if method == 'convolution':
        # frequency-domain gating
        kernel = fftshift(fft(ifftshift(gate), norm='forward'))
        ntwk_gated.s[:, 0, 0] = convolve1d(ntwk_gated.s[:, 0, 0], kernel, mode=conv_mode)
    elif method == 'fft':
        # time-domain band-pass mode
        s_td = fftshift(ifft(ntwk_gated.s[:, 0, 0]))
        s_td_g = s_td * gate
        ntwk_gated.s[:, 0, 0] = fft(ifftshift(s_td_g))
    elif method == 'rfft':
        # time-domain low-pass mode
        s_td = fftshift(irfft(ntwk_gated.s[:, 0, 0], n=len(t)))
        s_td_g = s_td * gate
        ntwk_gated.s[:, 0, 0] = rfft(ifftshift(s_td_g))

    # remove frequency-domain window
    ntwk_gated.s[:, 0, 0] = ntwk_gated.s[:, 0, 0] / window_fd

    if mode == 'bandstop':
        ntwk_gated = ntwk - ntwk_gated
    elif mode == 'bandpass':
        pass
    else:
        raise ValueError('mode should be \'bandpass\' or \'bandstop\'')

    return ntwk_gated
