"""
.. module:: skrf.time
========================================
time (:mod:`skrf.time`)
========================================

time domain functions 

Methods
------------

.. autosummary::
    :toctree: generated/

    time_gate
    detect_span 
    find_n_peaks
    indexes
    

    
"""

from .util import  find_nearest_index


from scipy.ndimage.filters import convolve1d
from scipy import  signal
import numpy as npy
import numpy as np # so i dont have to change indexes (from peakutils)
from numpy import fft

def indexes(y, thres=0.3, min_dist=1):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    
    Notes
    --------
    This function was taken from peakutils-1.1.0 
    http://pythonhosted.org/PeakUtils/index.html
    
    """
    #This function  was taken from peakutils, and is covered 
    # by the MIT license, inlcuded below: 
    
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
    
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)
    
    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])
    
    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

def find_n_peaks(x,n, thres=.9, **kw):
    '''
    Find a given number of peaks in a signal
    '''
    
    for dummy  in range(10):
        
        idx = indexes(x, **kw)
        if len(idx) < n:
            thres*=.5
            
        else:
            peak_vals = sorted(x[idx], reverse=True)[:n]
            peak_idxs =[x.tolist().index(k) for k in peak_vals]

            return peak_idxs
    raise ValueError('Couldnt find %i peaks'%n)
    
def detect_span(ntwk):
    '''
    detect the correct time-span between two largest peaks
    '''
    x = ntwk.s_time_db.flatten()
    p1,p2 = find_n_peaks(x,n=2)
    #distance to nearest neighbor peak
    span = abs(ntwk.frequency.t_ns[p1]-ntwk.frequency.t_ns[p2])
    return span 
    
def time_gate(ntwk, start=None, stop=None, center=None, span=None,
              mode='bandpass', window=('kaiser', 6),media=None, 
              boundary='reflect',return_all=False):
    '''
    Time-gate one-port s-parameters.
    
    The gate can be defined with start/stop times, or by 
    center/span. all times are in units of nanoseconds. common 
    windows are:
     * ('kaiser', 6)
     * 6 # integers are interpreted as kaiser beta-values
     * 'hamming'
     * 'boxcar'  # a staightup rect
     
    If no parameters are passed this will try to auto-gate the largest
    peak. 

    Parameters
    ------------
    start : number, or None
        start of time gate, (ns). 
    stop : number, or None
        stop of time gate (ns). 
    center : number, or None
        center of time gate, (ns). If None, and span is given, 
        the gate will be centered on the peak.
    span : number, or None
        span of time gate, (ns).  If None span will be half of the 
        distance to the second tallest peak
    mode: ['bandpass','bandstop']
        mode of gate 
    boundary:  {'reflect', 'constant', 'nearest', 'mirror', 'wrap'},
        passed to `scipy.ndimage.filters.convolve1d`
    window : string, float, or tuple 
        passed to `window` arg of `scipy.signal.get_window`
    
    Notes
    ------
    You cant gate things that are 'behind' strong reflections. This 
    is due to the multiple reflections that occur. 
    
    If `center!=0`, then the ntwk's time response is shifted 
    to t=0, gated, then shifted back to where it was. This is 
    done in frequency domain using `ntwk.delay()`. If the media being
    gated is dispersive (ie waveguide), then the gate `span` will be 
    span at t=0, which is different.
    
    If you need to time-gate an N-port network, then you should 
    gate each s-parameter independently. 
    
    Returns
    --------
    ntwk : Network
        copy of ntwk with time-gated s-parameters

    .. warning::
        Depending on sharpness of the gate, the  band edges may be 
        inaccurate, due to properties of FFT. We do not re-normalize
        anything.


    '''
    if ntwk.nports >1:
        raise ValueError('Time-gating only works on one-ports. try taking ntwk.s11 or ntwk.s21 first')

    if start is not None and stop is not None:
        start *= 1e-9
        stop *= 1e-9
        span = abs(stop-start)
        center = (stop-start)/2.
    
    else:
        if center is None:    
            # they didnt provide center, so find the peak
            n = ntwk.s_time_mag.argmax()
            center = ntwk.frequency.t_ns[n]
            
        if span is None:
            span = detect_span(ntwk)
            
        center *= 1e-9
        span *= 1e-9
        start = center - span / 2.
        stop = center + span / 2.

    
    # find start/stop gate indecies
    t = ntwk.frequency.t
    start_idx = find_nearest_index(t, start)
    stop_idx = find_nearest_index(t, stop)

    # create window
    window_width = abs(stop_idx - start_idx)
    window = signal.get_window(window, window_width)

    # create the gate by padding the window with zeros
    gate = npy.r_[npy.zeros(start_idx),
                           window,
                           npy.zeros(len(t) - stop_idx)]

    #IFFT the gate, so we have it's frequency response, aka kernel
    kernel=fft.fftshift(fft.ifft(fft.fftshift(gate, axes=0), axis=0))
    kernel =abs(kernel).flatten() # take mag and flatten
    kernel=kernel/sum(kernel) # normalize kernel
    
    out = ntwk.copy()
    
    # conditionally delay ntwk, to center at t=0, this is 
    # equivalent to gating at center.  (this is probably very inefficient)
    if center!=0:
        out = out.delay(-center*1e9, 'ns',port=0,media=media)
    
    # waste of code to handle convolve1d suck
    re = out.s_re[:,0,0]
    im = out.s_im[:,0,0]
    s = convolve1d(re,kernel, mode=boundary)+\
     1j*convolve1d(im,kernel, mode=boundary)
    out.s[:,0,0] = s
    # conditionally  un-delay ntwk
    if center!=0:
        out = out.delay(center*1e9, 'ns',port=0,media=media)

    if mode == 'bandstop':
        out = ntwk-out
    elif mode=='bandpass':
        pass
    else:
        raise ValueError('mode should be \'bandpass\' or \'bandstop\'')

    if return_all:
        # compute the gate ntwk and add delay
        gate_ntwk = out.s11.copy()
        gate_ntwk.s = kernel
        gate_ntwk= gate_ntwk.delay(center*1e9, 'ns', media=media)
        
        return {'gated_ntwk':out,
                'gate':gate_ntwk}
    else:
        return out
    



