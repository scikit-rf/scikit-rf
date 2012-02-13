import pylab
import skrf as rf

# create a Network type from a touchstone file of a horn antenna
horn = rf.Network('horn antenna.s1p')

# plot magnitude (in db) of S11
pylab.figure(1)
pylab.title('Return Loss (Mag)')
horn.plot_s_db(m=0,n=0) # m,n are S-Matrix indecies


# show the plots
pylab.show()
