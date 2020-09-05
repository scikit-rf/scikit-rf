import skrf as rf
import os

test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
a = rf.Network(os.path.join(test_dir,'ntwk_noise.s2p'))

nf = 10**(0.05)
b = rf.Network(f=[1, 2],
               s=[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
               z0=50).interpolate(a.frequency)
c = a ** b
r1 = c.nfmin[0]
d = b ** a
r2 = d.nfmin[0]
print('done')