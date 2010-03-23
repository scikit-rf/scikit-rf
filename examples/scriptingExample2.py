import mwavepy as m
from pylab import * 
from scipy.constants import inch, micron

#reload(m )
#close('all')
showPlot=True
meas = m.loadAllTouchstonesInDir()
[meas[k].changeFreqUnit('GHz') for k in meas]

frac = {}
for k in ['ewshort01','qwshort01']:
	frac[k+'/short'] = meas[k].divide(meas['short01'])
	

############################ CALIBRATION

## intermediate steps to define waveguide band
points = len(meas['short01'].freq)
mywg = m.WR1p5
centerLambda = mywg.lambdaG_f(mywg.fCenter) # ww = whole wave
freq =  meas['short01'].freq

## calculate ideal responses, put them in a dictionary
ideal = {}
ideal['qw'] = mywg.createDelayShort(l = centerLambda/4.,\
	numPoints = points,name='ideal qw')
ideal['ew'] = mywg.createDelayShort(l = centerLambda/8.,\
	numPoints = points, name='ideal ew')
ideal['oneInch'] = mywg.createDelayShort(l = 1*inch,\
	numPoints = points, name='ideal 1\"')
ideal['short'] = mywg.createShort(numPoints = points, \
	name='ideal short')
ideal['match'] = mywg.createMatch(numPoints = points, name='ideal match')
ideal['open'] = m.createNtwkFromTouchstone(\
'/home/alex/darpaProbe/simulation/rectangularApertureInFreeSPace/reactangularAPerture wr1.5.s1p', )
ideal['open'].name = 'ideal open'


# calculate first tier calibration
gammaMList = [meas[k] for k in ['short01','ewshort01','match01']]
gammaAList = [ideal[k] for k in ['short','ew','match']]
abc,residues = m.getABCLeastSquares(gammaMList = gammaMList,\
	gammaAList= gammaAList)
vnaErrorNtwk = m.abc2Ntwk(abc, isReciprocal=True, \
	name='vnaError', freq = freq,freqMultiplier=1e9)
vnaErrorCoefs = m.abc2CoefsDict(abc)

cal={}
for k in meas.keys():
	cal[k] = meas[k]/vnaErrorNtwk


if showPlot:
	figure()  
	cal['cpwshort01'].plotdB()
	cal['emptygold01'].plotdB()
	title('Comparison of Empty Block and Block with Loaded Probe')

	figure()  
	title('Phase Difference Between Empty Block and Two Difference Loads')
	cal['emptygold01'].divide(cal['cpwshort01']).plotPhase(unwrap=True)
	cal['emptygold01'].divide(cal['cpwshortB01']).plotPhase(unwrap=True)

	figure()
	title('Phase Difference Between Two Loads')
	cal['cpwshort01'].divide(cal['cpwshortB01']).plotPhase(unwrap=True)

show()
