import numpy as npy


def nfmin(gm, beta0, rbb, w, wT):
    yy = 1/beta0 + 1/beta0**2
    xx = yy + (w/wT)**2
    aa = 1 + 1/beta0

    return aa + gm*rbb*xx + npy.sqrt(yy + 2*gm*rbb*aa*xx + (gm*rbb)**2*xx**2)

def nf(z0, gm, beta0, rbb, w, wT):
    y0 = 1/z0
    yy = 1/beta0 + 1/beta0**2
    xx = (1 + npy.imag(y0)**2/(npy.real(y0)**2))

    aa = 1 + 1/beta0
    bb = 1 + rbb*npy.real(y0)*xx
    cc = npy.real(y0)/(gm*2)*xx
    dd = gm/2*yy
    ee = 2*rbb + 1/npy.real(y0) + rbb**2*npy.real(y0)*xx
    ff = npy.imag(y0)/npy.real(y0)*w/wT

    return aa*bb + cc + dd*ee + ff + gm/2*ee*(w/wT)**2

    