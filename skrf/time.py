"""
.. module:: skrf.time
========================================
time (:mod:`skrf.time`)
========================================

time domain functions 

Methods
===============

.. autosummary::
    :toctree: generated/

    time_gate
    detect_span 
    find_n_peaks
"""

from .util import  find_nearest_index

import peakutils
from scipy.ndimage.filters import convolve1d
from scipy import  signal
import numpy as npy
from numpy import fft

def find_n_peaks(x,n, thres=.9, **kw):
    '''
    Find a given number of peaks in a signal
    '''
    
    for dummy  in range(10):
        
        indexes = peakutils.indexes(x, **kw)
        if len(indexes) < n:
            thres*=.5
            
        else:
            peak_vals = sorted(x[indexes], reverse=True)[:n]
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

    if mode == 'bandstop':
        gate = 1 - gate

    #IFFT the gate, so we have it's frequency response, aka kernel
    kernel=fft.fftshift(fft.ifft(fft.fftshift(gate, axes=0), axis=0))
    kernel =abs(kernel).flatten() # take mag and flatten
    kernel=kernel/sum(kernel) # normalize kernel
    
    out = ntwk.copy()
    
    # conditionally delay ntwk, to center at t=0, this is 
    # equivalent to gating at center. 
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

    if return_all:
        # compute the gate ntwk and add delay
        gate_ntwk = out.s11.copy()
        gate_ntwk.s = kernel
        gate_ntwk= gate_ntwk.delay(center*1e9, 'ns', media=media)
        
        return {'gated_ntwk':out,
                'gate_ntwk':gate_ntwk}
    else:
        return out
    



