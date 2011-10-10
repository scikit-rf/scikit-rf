.. _example_oneport_calibration:
One-Port Calibration
***************

Instructive
--------------


This example is written to be instructive, not concise.::

	import mwavepy as mv
	
	
	## created necessary data for Calibration class
	
	# a list of Network types, holding 'ideal' responses
	my_ideals = [\
	        mv.Network('ideal/short.s1p'),
	        mv.Network('ideal/open.s1p'),
	        mv.Network('ideal/load.s1p'),
	        ]
	
	# a list of Network types, holding 'measured' responses
	my_measured = [\
	        mv.Network('measured/short.s1p'),
	        mv.Network('measured/open.s1p'),
	        mv.Network('measured/load.s1p'),
	        ]
	
	## create a Calibration instance
	cal = mv.Calibration(\
	        ideals = my_ideals,
	        measured = my_measured,
	        )
	
	
	## run, and apply calibration to a DUT
	
	# run calibration algorithm
	cal.run() 
	
	# apply it to a dut
	dut = mv.Network('my_dut.s1p')
	dut_caled = cal.apply_cal(dut)
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

Concise
-----------

This example is meant to be the same as the first except more concise::

	import mwavepy as mv
	
	my_ideals = mv.load_all_touchstones_in_dir('ideals/')
	my_measured = mv.load_all_touchstones_in_dir('measured/')
	
	
	## create a Calibration instance
	cal = mv.Calibration(\
	        ideals = [my_ideals[k] for k in ['short','open','load']],
	        measured = [my_measured[k] for k in ['short','open','load']],
	        )
	
	## what you do with 'cal' may  may be similar to above example

