.. _example_twoport_calibration
Two-Port Calibration
***********************



This is an example of how to setup two-port calibration. For more detailed explaination see :doc:`calibration`::
	
	import skrf as rf
	
	
	## created necessary data for Calibration class
	
	# a list of Network types, holding 'ideal' responses
	my_ideals = [\
	        rf.Network('ideal/thru.s2p'),
	        rf.Network('ideal/line.s2p'),
	        rf.Network('ideal/short, short.s2p'),
	        ]
	
	# a list of Network types, holding 'measured' responses
	my_measured = [\
	        rf.Network('measured/thru.s2p'),
	        rf.Network('measured/line.s2p'),
	        rf.Network('measured/short, short.s2p'),
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
	dut = rf.Network('my_dut.s2p')
	dut_caled = cal.apply_cal(dut)
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

