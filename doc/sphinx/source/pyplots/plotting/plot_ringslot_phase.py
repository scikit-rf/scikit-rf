import pylab
import mwavepy as mv 

ring_slot= mv.Network('ring slot array measured.s1p')
pylab.figure(1)
pylab.title('WR-10 Ringslot Array, Phase')
# kwargs given to plot commands are passed through to the pylab.plot 
# command
ring_slot.plot_s_deg(m=0,n=0, color='r', markevery=5, marker='o')
pylab.show()
