import mwavepy as mv
from pylab import * 

'''
a simple example of how to make and  over-determined one-port
calibration, apply calibration to a measurement, and then save results
'''

# setup measured and ideal lists (ORDER MATTERS)
measured = [\
	mv.Network('measured/ds1.s1p'),
	mv.Network('measured/ds2.s1p'),
	mv.Network('measured/ds3.s1p'),
	mv.Network('measured/ds4.s1p'),
	mv.Network('measured/ds5.s1p'),
	]
	
ideals = [\
	mv.Network('ideals/ds1.s1p'),
	mv.Network('ideals/ds2.s1p'),
	mv.Network('ideals/ds3.s1p'),
	mv.Network('ideals/ds4.s1p'),
	mv.Network('ideals/ds5.s1p'),
	]

# initialize Calibration 
cal = mv.Calibration(measured = measured, ideals = ideals)

# run calibration algorithm
cal.run()



#save error network 
cal.error_ntwk.write_touchstone('error_network.s2p')

# apply calibration to a measurement 
ds1_caled = cal.apply_cal(mv.Network('measured/ds1.s1p'))

#save calibrated data to file
ds1_caled.write_touchstone('caled/ds1.s1p')


# plot calibration error parameters
figure(1)
cal.plot_coefs_db()

#plots calibrated data contents
figure(2)
ds1_caled.plot_s_smith()

draw();show()
