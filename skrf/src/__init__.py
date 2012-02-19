
#       connect.py
#
#       Copyright 2012 alex arsenovic <arsenovic@virginia.edu>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inct., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import numpy as npy
import ctypes as ct
import os

src_path = os.path.dirname(__file__)
connect_lib = npy.ctypeslib.load_library('libconnect.so.1.0.1', src_path)

def connect_s_fast(A,k,B,l):
    '''
    connect two n-port networks' s-matricies together.

    specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matricies. The function
    :func:`connect` operates on :class:`Network` types.

    Parameters
    -----------
    A : numpy.ndarray
            S-parameter matrix of `A`, shape is fxnxn
    k : int
            port index on `A` (port indecies start from 0)
    B : numpy.ndarray
            S-parameter matrix of `B`, shape is fxnxn
    l : int
            port index on `B`

    Returns
    -------
    C : numpy.ndarray
            new S-parameter matrix


    Notes
    -------
    internally, this function creates a larger composite network
    and calls the  :func:`innerconnect_s` function. see that function for more
    details about the implementation

    See Also
    --------
            connect : operates on :class:`Network` types
            innerconnect_s : function which implements the connection
                    connection algorithm


    '''
    if k > A.shape[-1]-1 or l>B.shape[-1]-1:
        raise(ValueError('port indecies are out of range'))

    freq = npy.ones(len(A)) 
    nFreq = len (freq)
    nA = A.shape[2]
    nB = B.shape[2]
    C = B.copy()
    nC = nA+nB-2
    connect_lib.connect_s(
        freq.ctypes.data_as(ct.POINTER(ct.c_float)), 
        nFreq,
        A.ctypes.data_as(ct.POINTER(ct.c_float)),
        nA,
        k,
        B.ctypes.data_as(ct.POINTER(ct.c_float)),
        nB,
        l,
        C.ctypes.data_as(ct.POINTER(ct.c_float)),
        nC)
    return C
