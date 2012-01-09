import pylab
import mwavepy as mv 

ring_slot= mv.Network('ring slot array measured.s1p')

pylab.figure(1)
pylab.title('WR-10 Ringslot Array, Smith')
ring_slot.plot_s_smith(m=0,n=0) # m,n are S-Matrix indecies
pylab.show()
