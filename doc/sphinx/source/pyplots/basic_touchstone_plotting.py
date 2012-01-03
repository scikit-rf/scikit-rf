import pylab
import mwavepy as mv 


dir = 'basic_touchstone_plotting/'
# create a Network type from a touchstone file of a horn antenna
horn = mv.Network(dir+'horn_antenna.s1p')

# plot magnitude of S11
pylab.figure(1)
pylab.title('Return Loss (Mag)')
horn.plot_s_db(m=0,n=0) # m,n are S-Matrix indecies
# show the plots (only needed if you dont have interactive set on ipython)
pylab.show()
