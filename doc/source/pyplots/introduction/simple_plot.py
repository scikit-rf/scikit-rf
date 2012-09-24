import pylab
import skrf as rf

# create a Network type from a touchstone file
ring_slot = rf.Network('ring slot.s2p')
ring_slot.plot_s_smith()
pylab.show()
