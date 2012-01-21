.. _example_oneport_calibration:
One-Port Calibration
***************

Instructive
--------------


This example is written to be instructive, not concise.::

	import skrf as rf
	
	
	## created necessary data for Calibration class
	
	# a list of Network types, holding 'ideal' responses
	my_ideals = [\
	        rf.Network('ideal/short.s1p'),
	        rf.Network('ideal/open.s1p'),
	        rf.Network('ideal/load.s1p'),
	        ]
	
	# a list of Network types, holding 'measured' responses
	my_measured = [\
	        rf.Network('measured/short.s1p'),
	        rf.Network('measured/open.s1p'),
	        rf.Network('measured/load.s1p'),
	        ]
	
	## create a Calibration instance
	cal = rf.Calibration(\
	        ideals = my_ideals,
	        measured = my_measured,
	        )
	
	
	## run, and apply calibration to a DUT
	
	# run calibration algorithm
	cal.run() 
	
	# apply it to a dut
	dut = rf.Network('my_dut.s1p')
	dut_caled = cal.apply_cal(dut)
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

Concise
-----------

This example is meant to be the same as the first except more concise::

	import skrf as rf
	
	my_ideals = rf.load_all_touchstones_in_dir('ideals/')
	my_measured = rf.load_all_touchstones_in_dir('measured/')
	
	
	## create a Calibration instance
	cal = rf.Calibration(\
	        ideals = [my_ideals[k] for k in ['short','open','load']],
	        measured = [my_measured[k] for k in ['short','open','load']],
	        )
	
	## what you do with 'cal' may  may be similar to above example

