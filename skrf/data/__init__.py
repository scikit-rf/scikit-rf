
'''
.. module:: skrf.data
========================================
io (:mod:`skrf.data`)
========================================


This Package provides data to be used in examples and testcases

Modules
----------
.. toctree::
   :maxdepth: 1

   
'''
import os 

from  ..network import Network
from ..io.general import read

pwd = os.path.dirname(os.path.abspath(__file__))

ntwk1 = Network(os.path.join(pwd, 'ntwk1.s2p'))
line = Network(os.path.join(pwd, 'line.s2p'))
ring_slot = Network(os.path.join(pwd, 'ring slot.s2p'))
ring_slot_meas = Network(os.path.join(pwd, 'ring slot measured.s1p'))
wr2p2_line = Network(os.path.join(pwd, 'wr2p2,line.ntwk'))
wr2p2_line1 = Network(os.path.join(pwd, 'wr2p2,line1.ntwk'))
wr2p2_delayshort = Network(os.path.join(pwd, 'wr2p2,delayshort.ntwk'))
wr2p2_short = Network(os.path.join(pwd, 'wr2p2,short.ntwk'))
wr1p5_line = Network(os.path.join(pwd, 'wr1p5,line.ntwk'))
wr1p5_short = Network(os.path.join(pwd, 'wr1p5,short.ntwk'))

one_port_cal = read(os.path.join(pwd, 'one_port.cal'))

mpl_rc_fname = os.path.join(pwd, 'skrf.mplstyle')

## material database (taken from wikipedia)
materials = {
    'copper':{
        'resistivity(ohm*m)':1.68e-8,
        },
    'aluminum':{
        'resistivity(ohm*m)':2.82e-8,
        },
    'gold':{
        'resistivity(ohm*m)':2.44e-8,
        },
    'mylar':{
        'relative permativity':3.1,
        'loss tangent':500e-4,
        },
    'quartz':{
        'relative permativity':3.8,
        'loss tangent':1.5e-4,
        },
    'silicon':{
        'relative permativity':11.68,
        'loss tangent':8e-4,
        },
    'teflon':{
        'relative permativity':2.1,
        'loss tangent':5e-4,
        },
    'duroid 5880':{
        'relative permativity':2.25,
        'loss tangent':40e-4,
        },
    }
for k1,k2 in [
    ('cu', 'copper'), 
    ('al', 'aluminum'),
    ('au', 'gold')]:
    materials[k1] = materials[k2]
