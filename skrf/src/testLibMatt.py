import skrf as mv
import numpy as np
import ctypes as C

a = mv.Network("Probe.s2p")
b = mv.Network("Probe.s2p")


matt = np.ctypeslib.load_library('libconnect.so.1.0.1', '.')


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


def foo2():
    mvResult = mv.connect_s(A, k, B, l)



def foo():
    matt.connect_s(freq.ctypes.data_as(C.POINTER(C.c_float)), nFreq, A.ctypes.data_as(C.POINTER(C.c_float)), nA, k, B.ctypes.data_as(C.POINTER(C.c_float)), nB, l, mattResult.ctypes.data_as(C.POINTER(C.c_float)), nResult)


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

