import skrf as rf
from pylab import *

cal = rf.Calibration(\
        measured = [\
                rf.Network('measured/short.s1p'),
                rf.Network('measured/delay short 132um.s1p'),
                rf.Network('measured/delay short 85um.s1p'),
                rf.Network('measured/load.s1p'),
                ],
        ideals =[\
                rf.Network('ideals/short.s1p'),
                rf.Network('ideals/delay short 132um.s1p'),
                rf.Network('ideals/delay short 85um.s1p'),
                rf.Network('ideals/load.s1p'),
                ],
        )

ro_meas = rf.Network('dut/radiating open.s1p')
ro_cal = cal.apply_cal(ro_meas)

ro_sim = rf.Network('simulation/radiating open.s1p')

figure()
ro_cal.plot_s_db(label='Experiment')
ro_sim.plot_s_db(label='Simulated')

draw();show();
