'''
records a series of sweeps from a vna, to touchstone files

this was used to characterize the noise of a vna.


'''
import sys
sys.path.append('../../')
import mwavepy as mv
import os,datetime

nsweeps = 101 # number of sweeps to take
dir = 'touchstoneFiles2' # directory to save files in 

myvna = mv.vna.HP8720() # HP8510 also available

for k in range(nsweeps):
	print  k 
	ntwk = myvna.s11
	date_string = datetime.datetime.now().__str__().replace(':','-')
	ntwk.write_touchstone(dir +'/'+ date_string)
	
