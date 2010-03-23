import sys
sys.path.append('../')
import mwavepy as m
import pylab 

# this should be accessed from some material database
conductivityCopper=5.7e7 #S/m

pylab.figure()
# loops through waveguide types, and plots the conductor loss (alphaC) 
# vs frequency

for k in [m.wr(90), m.wr(62), m.wr(42),m.wr(28), m.wr(19),m.wr(15)]:
	pylab.plot(k.freqAxis/1e9, k.alphaC(k.freqAxis*(2*pylab.pi), \
		conductivity =conductivityCopper), label= k.name) 

pylab.legend()
pylab.xlabel('Frequenc (GHz)')
pylab.ylabel ('Loss (dB/m)')
pylab.title('Waveguide Loss, for Copper ($\sigma=5.7E7$)')


pylab.savefig('waveguideLoss.eps',format='eps')
pylab.savefig('waveguideLoss.png',format='png')


pylab.show()
