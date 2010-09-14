
'''
#       calibrationAlgorithms.py
#       
#       
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#       
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
'''
from copy import copy
from calibrationFunctions import * 
import numpy as npy
from scipy.optimize import fmin 
def one_port(measured, ideals):
	
	'''
	standard algorithm for a one port calibration. If more than three 
	standards are supplied then a least square algorithm is applied.
	 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
		ideals - list of assumed reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray, representing a 
			s-matrix or list of  1-port mwavepy.ntwk types. 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a = abc[:,0] = e01*e10 - e00*e11
				b = abc[:,1] = e00
				c = abc[:,2] = e11
			
			residuals: a matrix of residuals from the least squared 
				calculation. see numpy.linalg.lstsq() for more info	
	'''
	#make  copies so list entities are not changed, when we typecast 
	mList = copy(measured)
	iList = copy(ideals)
	
	numStds = len(mList)# find number of standards given, for dimensions
	numCoefs=3
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			mList[k] = mList[k].s.reshape((-1,1))
			iList[k] = iList[k].s.reshape((-1,1))
	except:
		pass	
	
	# ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
	fLength = len(mList[0])
	
	#initialize outputs 
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residuals = npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 
	
	# loop through frequencies and form m, a vectors and 
	# the matrix M. where M = 	i1, 1, i1*m1 
	#							i2, 1, i2*m2
	#									...etc
	for f in range(fLength):
		#create  m, i, and 1 vectors
		one = npy.ones(shape=(numStds,1))
		m = npy.array([ mList[k][f] for k in range(numStds)]).reshape(-1,1)# m-vector at f
		i = npy.array([ iList[k][f] for k in range(numStds)]).reshape(-1,1)# i-vector at f			
		
		# construct the matrix 
		Q = npy.hstack([i, one, i*m])
		# calculate least squares
		abcTmp, residualsTmp = npy.linalg.lstsq(Q,m)[0:2]
		#if len (residualsTmp )==0:
		#	raise ValueError( 'matrix has singular values, check standards')
			
		abc[f,:] = abcTmp.flatten()
		residuals[f,:] = residualsTmp

	

	# output is a dictionary of information
	output = {'error coefficients':abc_2_coefs_dict(abc), 'residuals':residuals}
	
	return output


def xds(measured, ideals, wb, d, ftol=1e-3, xtol=1e-3, \
	guessLength=False,solveForLoss=False,showProgress= False):
	'''
	A one port calibration, which can use a redundent number of delayed 
	shorts to solve	for their unknown lengths.
	
	!see note at bottom about order!
	 
	takes: 
		measured - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray. 
		ideals - list of measured reflection coefficients. can be 
			lists of either a kxnxn numpy.ndarray. see note about order.
		wb - a mwavepy.workingBand.WorkingBand type. 
		d - vector containing initial guesses for the delay short lengths
			see note about order.
		ftol - functional tolerance, passed to the scipy.optimize.fmin 
			function
		solveForLoss - 
		guessLength - 
		showProgress - 
	
	returns:
		(abc, residues) - a tuple. abc is a Nx3 ndarray containing the
			complex calibrations coefficients,where N is the number 
			of frequency points in the standards that where given.
			
			abc: 
			the components of abc are 
				a[:] = abc[:,0]
				b[:] = abc[:,1]
				c[:] = abc[:,2],
			a, b and c are related to the error network by 
				a = e01*e10 - e00*e11 
				b = e00 
				c = e11
			
			residues: a matrix of residues from the least squared 
				calculation. see numpy.linalg.lstsq() for more info
	
	 
		
	 note:
		ORDER MATTERS.
	
		all standard lists, and d-vector must be in order. The first
		m-standards are assumed to be delayed shorts, where m is the
		 length of d. Any standards after may be anything.
	
	'''
	
	
	
	
	#make deep copies so list entities are not changed
	gammaMList = copy(measured)
	gammaAList = copy(ideals)
	d = copy(d)
	d = list(d)
	
	
		
	# find number of standards given, set numberCoefs. Used for matrix 
	# dimensions
	numStds = len(gammaMList)
	numCoefs = 3
	if solveForLoss == True:
		numDelays = len(d)-1
	else:
		numDelays = len(d)
		
	
	if guessLength == True:
		#making assumption first non-delay network is teh Short
		short = gammaMList[numDelays]
		for k in range(numDelays):
			d[k] = findPhysicalLengthOfDelayShort(gammaMList[k]/short, wb) 
		
		print array(d)/1e-6
		
		
	# try to access s-parameters, in case its a ntwk type, other wise 
	# just keep on rollin 
	try:
		for k in range(numStds):
			gammaMList[k] = gammaMList[k].s
			gammaAList[k] = gammaAList[k].s
	
	except:
		pass	
	
	
	
	fLength = len(gammaMList[0])
	#initialize output 
	abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex) 
	residues = npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex) 

	
	
	if solveForLoss == True:
		if wb.surfaceConductivity is None:
			# they have supplied us with a surface Conductivity, default to Al
			wb.surfaceConductivity = conductivityDict['alumninium']
		
		# append conductivity to the d-list, this is sloppy, but i dont
		# see a way around it
		d.append(wb.surfaceConductivity)
		
		def iterativeCal(d, gammaMList, gammaAList):
			#TODO:  this function uses sloppy namespace, which limits portability
			#pick off last element of d, which will hold the conductivity
			wb.surfaceConductivity = d[-1]
			numDelays = len(d)-1
			
			for stdNum in range(numDelays):
				gammaAList[stdNum] = wb.delay_short(d = d[stdNum]).s
			
			abc, residues = one_port(gammaMList, gammaAList)
			sumResidualList.append(npy.sum(abs(residues)))
			#print npy.sum(abs(residues))
			
			if showProgress == True:
				print npy.sum(abs(residues)),'==>',npy.linalg.linalg.norm(d),d
			return npy.sum(abs(residues))	
			
	else:
		def iterativeCal(d, gammaMList, gammaAList):
			#TODO:  this function uses sloppy namespace, which limits portability
			numDelays=len(d)
			for stdNum in range(numDelays):
				gammaAList[stdNum] = wb.delay_short(d = d[stdNum]).s

			
			residues = one_port(gammaMList, gammaAList)['residuals']
			sumResidualList.append(npy.sum(abs(residues)))
			#print npy.sum(abs(residues))
			if showProgress == True:
				print npy.sum(abs(residues)),'==>',npy.linalg.linalg.norm(d),d
			return npy.sum(abs(residues))
	
	
	dStart = npy.array(d)
	sumResidualList = []	
	
	dEnd = fmin (iterativeCal, dStart,args=(gammaMList,gammaAList), \
		disp=False,ftol=ftol, xtol=xtol)
	
	if solveForLoss == True:
		wb.surfaceConductivity = dEnd[-1]
		
		
	for stdNum in range(numDelays):
			gammaAList[stdNum] = wb.delay_short(d = dEnd[stdNum]).s
			
		

	output = one_port (measured = gammaMList, ideals=gammaAList)


	output.update( {\
	'd_end':dEnd,\
	'sum_residual_list':sumResidualList\
	})
	return output
