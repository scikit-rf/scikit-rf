.. _example_twoport_calibration
Two-Port Calibration
***********************



This is an example of how to setup two-port calibration. For more detailed explaination see :doc:`calibration`::
	
	import mwavepy as mv
	
	
	## created necessary data for Calibration class
	
	# a list of Network types, holding 'ideal' responses
	my_ideals = [\
	        mv.Network('ideal/thru.s2p'),
	        mv.Network('ideal/line.s2p'),
	        mv.Network('ideal/short, short.s2p'),
	        ]
	
	# a list of Network types, holding 'measured' responses
	my_measured = [\
	        mv.Network('measured/thru.s2p'),
	        mv.Network('measured/line.s2p'),
	        mv.Network('measured/short, short.s2p'),
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
	dut = mv.Network('my_dut.s2p')
	dut_caled = cal.apply_cal(dut)
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

