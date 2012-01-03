import mwavepy as mv
from pylab import * 

cal = mv.Calibration(\
	measured = [\
		mv.Network('measured/short.s1p'),
		mv.Network('measured/delay short 132um.s1p'),
		mv.Network('measured/delay short 85um.s1p'),
		mv.Network('measured/load.s1p'),
		],
	ideals =[\
		mv.Network('ideals/short.s1p'),
		mv.Network('ideals/delay short 132um.s1p'),
		mv.Network('ideals/delay short 85um.s1p'),
		mv.Network('ideals/load.s1p'),
		],
	)

ro_meas = mv.Network('dut/radiating open.s1p')
ro_cal = cal.apply_cal(ro_meas)

ro_sim = mv.Network('simulation/radiating open.s1p')

figure()
ro_cal.plot_s_db(label='Experiment')
ro_sim.plot_s_db(label='Simulated')

draw();show();
