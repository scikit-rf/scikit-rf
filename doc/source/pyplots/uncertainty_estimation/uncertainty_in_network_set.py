import pylab
import skrf as rf

ro_set = rf.NetworkSet(\
        rf.load_all_touchstones('.',contains='ro').values(),\
        name = 'Radiating Open')

pylab.figure()
pylab.title('Uncertainty in Phase')
ro_set.plot_uncertainty_bounds_s_deg()

pylab.figure()
pylab.title('Uncertainty in Magnitude')
ro_set.plot_uncertainty_bounds_s_db()

pylab.show()
