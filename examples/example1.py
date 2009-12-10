#!/bin/python
import sys
sys.path.append('../')
import mwavepy as m 
import pylab as p 
#normally mwavepy would be placed in your path, so you can use import instead of __import__

# load the touchone file, which we got from the VNA
horn = m.loadTouchtone('horn.s2p')

#plot returnloss in dB, on a smith chart and, plot SWR.
p.figure(1)
horn.s11.plotdB()

p.figure(2)
horn.s11.plotSmith()
p.axis('equal')
p.axis([-1,1,-1,1])

p.figure(3)
horn.plotSwr1()
p.ylim(0,3)
p.show()
