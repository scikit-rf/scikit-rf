import pylab
import skrf as rf

horn = rf.Network('horn antenna.s1p')
pylab.title('WR-10 Horn Antenna')
horn.plot_s_deg(m=0,n=0,color='r',marker='o',markevery=5)
# kwargs are passed through to pylab.plot()
pylab.show()
