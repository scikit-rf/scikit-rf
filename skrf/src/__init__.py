

import numpy as npy
import ctypes as ct
import os
import platform

src_path = os.path.dirname(__file__)
if platform.system() == 'Windows':
    lib_name = 'connect.pyd'
else:
    lib_name = 'connect.so'
connect_lib = npy.ctypeslib.load_library(lib_name, src_path)

def connect_s_fast(A,k,B,l):
    '''
    connect two n-port networks' s-matrices together.

    specifically, connect port `k` on network `A` to port `l` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matrices. The function
    :func:`connect` operates on :class:`Network` types.

    Parameters
    -----------
    A : numpy.ndarray
            S-parameter matrix of `A`, shape is fxnxn
    k : int
            port index on `A` (port indices start from 0)
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
        raise(ValueError('port indices are out of range'))

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
