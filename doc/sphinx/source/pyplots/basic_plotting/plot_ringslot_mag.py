import pylab
import mwavepy as mv 

# create a Network type from a touchstone file of a horn antenna
ring_slot= mv.Network('ring slot array measured.s1p')

# plot magnitude (in db) of S11
pylab.figure(1)
pylab.title('WR-10 Ringslot Array, Mag')
ring_slot.plot_s_db(m=0,n=0) # m,n are S-Matrix indecies
pylab.figure(2)
ring_slot.plot_s_db(m=0,n=0) # m,n are S-Matrix indecies
# show the plots 
pylab.show()
