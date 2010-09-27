import sys
sys.path.append('../')
import mwavepy as m
import pylab


# load a touchstone file into a 'ntwk' type
myNtwk = m.createNtwkFromTouchstone('horn.s2p')

# plot the data in some different formats
pylab.figure(1)
myNtwk.plotdB(m=0,n=0)	# m,n are S-Matrix indecies
pylab.title('Return Loss (Mag)')	# all matploting functions can be accesed

pylab.figure(2)
myNtwk.plotPhase(0,0)
pylab.title('Return Loss (Phase)')

pylab.figure(3)
myNtwk.plotSmith(0,0)
pylab.title('Return Loss')





# saveing teh file as vector or raster formats
pylab.figure(3)
pylab.savefig('smithChart.eps',format='eps')
pylab.savefig('smithChart.png',format='png')


pylab.figure(1)
pylab.savefig('returnLoss.eps',format='eps')
pylab.savefig('returnLoss.png',format='png')


pylab.show()
