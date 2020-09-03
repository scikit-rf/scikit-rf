import skrf as rf
import numpy as npy
from skrf.constants import *

T0 = 290. # Both resistors are set at room temperature for now
frequency = rf.Frequency(start=1000, stop=2000, npoints=10, unit='MHz')

ovec = npy.ones(len(frequency))
zvec = npy.zeros(len(frequency))
rseries = 200*ovec # Series resistance
rshunt = 500*ovec  # Shunt resistance

# This is the Y form of the network parameters for a series resistor
r_series1_y = rf.network_array([[1/rseries,  -1/rseries],
                               [-1/rseries,  1/rseries]])

r_series2_y = rf.network_array([[1/rseries,  -1/rseries],
                               [-1/rseries,  1/rseries]])

ntwk1 = rf.Network.from_y(r_series1_y, frequency = frequency)
ntwk1.noise_source('passive')
ntwk2 = rf.Network.from_y(r_series1_y, frequency = frequency)
ntwk2.noise_source('passive')

ntwk3 = rf.parallel_parallel_2port(ntwk1, ntwk2)
ntwk3.y
ntwk3.cy