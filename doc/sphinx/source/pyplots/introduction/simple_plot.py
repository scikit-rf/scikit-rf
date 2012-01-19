import pylab
import skrf as mv 

# create a Network type from a touchstone file
ring_slot = mv.Network('ring slot.s2p')
ring_slot.plot_s_smith()
pylab.show()
