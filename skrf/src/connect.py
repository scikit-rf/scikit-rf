
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


import skrf as mv
import numpy as np
import ctypes as ct


connect_lib = np.ctypeslib.load_library('libconnect.so.1.0.1', '.')
a = mv.Network("Probe.s2p")
b = mv.Network("Probe.s2p")

freq = a.f
nFreq = a.f.shape[-1]

A = a.s
nA = 2
k = 1

B = b.s
nB = 2
l = 0

mattResult = B.copy()
mattResult = mattResult*0
nResult = 2

mvResult = B.copy()
mvResult = mvResult*0

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

    freq = np.ones(len(A)) 
    nFreq = len (freq)
    nA = A.shape[2]
    nB = B.shape[2]
    C = B.copy()
    
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
        nResult)
    return C
'''


def foo2():
    mvResult = mv.connect_s(A, k, B, l)



def foo():
    return matt.connect_s(freq.ctypes.data_as(ct.POINTER(ct.c_float)), nFreq, A.ctypes.data_as(ct.POINTER(ct.c_float)), nA, k, B.ctypes.data_as(ct.POINTER(ct.c_float)), nB, l, mattResult.ctypes.data_as(ct.POINTER(ct.c_float)), nResult)


import timeit
t = timeit.Timer(setup='from __main__ import foo', stmt='foo()')
t2 = timeit.Timer(setup='from __main__ import foo2', stmt='foo2()')

print t.timeit(10)
print t2.timeit(10)


mvResult = mv.connect_s(A, k, B, l)

print
print mattResult[0]
print
print mvResult[0]
print
print mattResult[201]
print
print mvResult[201]
print
print mattResult[400]
print
print mvResult[400]
print

'''
