"""
data (:mod:`skrf.data`)
========================================

This Package provides data to be used in examples and testcases.

Material Properties
-------------------
.. autosummary::
   :toctree: generated/

    materials

Example/Test Networks
---------------------
.. autosummary::
   :toctree: generated/

    ring_slot
    ring_slot_meas
    line
    tee

"""
import os

from ..network import Network
from ..io.general import read

pwd = os.path.dirname(os.path.abspath(__file__))

ntwk1 = Network(os.path.join(pwd, 'ntwk1.s2p'))
line = Network(os.path.join(pwd, 'line.s2p'))
""" 2-Port Network: 'line',  75.0-110.0 GHz, 201 pts, z0=[50.+0.j 50.+0.j]"""
ring_slot = Network(os.path.join(pwd, 'ring slot.s2p'))
""" 2-Port Network: 'ring slot',  75.0-110.0 GHz, 201 pts, z0=[50.+0.j 50.+0.j] """
tee = Network(os.path.join(pwd, 'tee.s3p'))
""" 3-Port Network: 'tee',  330.0-500.0 GHz, 201 pts, z0=[50.+0.j 50.+0.j 50.+0.j] """
ring_slot_meas = Network(os.path.join(pwd, 'ring slot measured.s1p'))
"""  1-Port Network: 'ring slot measured',  75.0-109.999999992 GHz, 101 pts, z0=[50.+0.j] """
wr2p2_line = Network(os.path.join(pwd, 'wr2p2,line.s2p'))
wr2p2_line1 = Network(os.path.join(pwd, 'wr2p2,line1.s2p'))
wr2p2_delayshort = Network(os.path.join(pwd, 'wr2p2,delayshort.s1p'))
wr2p2_short = Network(os.path.join(pwd, 'wr2p2,short.s1p'))
wr1p5_line = Network(os.path.join(pwd, 'wr1p5,line.s2p'))
wr1p5_short = Network(os.path.join(pwd, 'wr1p5,short.s1p'))
ro_1 = Network(os.path.join(pwd, 'ro,1.s1p'))
ro_2 = Network(os.path.join(pwd, 'ro,2.s1p'))
ro_3 = Network(os.path.join(pwd, 'ro,3.s1p'))

"""
The following networks have been saved using Python2 Pickler. 
Numpy arrays pickled in Python2 can't be reopened using Python3, 
as a consequence (?) of the utf8 support in Python3.
So the following lines shall only work with Python2.  
"""
one_port_cal = read(os.path.join(pwd, 'one_port.cal'), encoding='latin1')

mpl_rc_fname = os.path.join(pwd, 'skrf.mplstyle')

# material database (taken from wikipedia)
materials = {
    'copper': {
        'resistivity(ohm*m)': 1.68e-8,
    },
    'aluminum': {
        'resistivity(ohm*m)': 2.82e-8,
    },
    'gold': {
        'resistivity(ohm*m)': 2.44e-8,
    },
    'lead': {
        'resistivity(ohm*m)': 1/4.56e6,  # from pozar appendix F
    },
    'steel(stainless)': {
        'resistivity(ohm*m)': 1/1.1e6,  # from pozar appendix F
    },
    'mylar': {
        'relative permativity': 3.1,
        'loss tangent': 500e-4,
    },
    'quartz': {
        'relative permativity': 3.8,
        'loss tangent': 1.5e-4,
    },
    'silicon': {
        'relative permativity': 11.68,
        'loss tangent': 8e-4,
    },
    'teflon': {
        'relative permativity': 2.1,
        'loss tangent': 5e-4,
    },
    'duroid 5880': {
        'relative permativity': 2.25,
        'loss tangent': 40e-4,
    },
}
"""
materials : Dictionnary of materials properties
"""

for k1,k2 in [
    ('cu', 'copper'),
    ('al', 'aluminum'),
    ('au', 'gold')]:
    materials[k1] = materials[k2]
