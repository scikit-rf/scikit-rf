'''
initial testcases for some of skrf's functionality.
'''
import skrf as rf
from scipy.constants import mil
import pylab as plb
import numpy as npy
import pdb


def wg_characteristic_impedance():
    '''
    produce a plot of the normalized characteristic impedances for
    TE and TM waveguide modes, vs normalized frequency.

    impedance normalized to characteristic impedance of freespace
    frequency normalized to cut-off frequency of given mode


    This example was produces same figure 4-3 in Harrington's
    Time-Harmonic Electromagnetic Fields
    '''
    # create waveguide type
    wg = rf.transmissionLine.RectangularWaveguide(a=1)
    # calculate cut-off frequency
    fc = wg.f_c(1,0)
    # create frequency vector
    f= npy.linspace(0,3*fc,101)
    # calculate free-space impedance
    eta = rf.transmissionLine.FreeSpace().z0(1)

    # plot the characteristic impedances
    plb.plot(f/fc , wg.z0('te',1,0,f)/eta, label= 'R te')
    plb.plot(f/fc, npy.imag(wg.z0('te',1,0,f))/eta, label='X te')
    plb.plot(f/fc, wg.z0('tm',1,0,f)/eta, label='R tm')
    plb.plot(f/fc, -npy.imag(wg.z0('tm',1,0,f))/eta, label = '-X tm')

    #label and show
    plb.title('Rectangular Waveguide TE10 Normalized Characterisitic Impedance')
    plb.ylabel('Normalized Impedance (eta/eta0)')
    plb.xlabel('Normalized Frequency (f/fc)')
    plb.axis('tight')
    plb.ylim(0,2)
    plb.legend()
    plb.show()
    return 1

def wg_propagation_constant():
    '''
    plot the normalized propagation constant for TE10 waveguide mode,
    vs normalized frequency.

    propagation constant normalized to freespace propagation constant
    at cut-off frequency of teh TE10 mode.
    frequency normalized to cut-off frequency of given mode


    This example was produces same figure 2-18 in Harrington's
    Time-Harmonic Electromagnetic Fields
    '''
    wg = rf.transmissionLine.RectangularWaveguide(a=1)
    free_space = rf.transmissionLine.FreeSpace()
    #pdb.set_trace()
    fc = wg.f_c(1,0)
    f= npy.linspace(0,5*fc,101)
    gamma0 = -1*free_space.propagation_constant(fc)

    plb.plot(f/fc , -1*(wg.propagation_constant(1,0,f))/gamma0, \
            label='waveguide: -Re(gamma)')
    plb.plot(f/fc , npy.imag(wg.propagation_constant(1,0,f))/gamma0,\
            label='waveguide: Im(gamma)')
    plb.plot(f/fc , -1*(free_space.propagation_constant(f))/gamma0,\
            label='freespace: -Re (gamma)')

    plb.title('Rectangular Waveguide TE10 Normalized Propagation Constant')
    plb.ylabel('Normalized Propagation Constant')
    plb.xlabel('Normalized Frequency [f/fc]')
    plb.axis('tight')
    #plb.ylim(0,2)
    plb.legend()
    plb.show()
