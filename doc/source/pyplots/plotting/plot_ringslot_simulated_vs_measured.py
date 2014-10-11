import pylab
import skrf as rf

# from the extension you know this is a 2-port network
ring_slot_sim = rf.Network('ring slot array simulation.s2p')
ring_slot_meas = rf.Network('ring slot array measured.s1p')

pylab.figure(1)
pylab.title('WR-10 Ringslot Array Simulated vs Measured')
# if no indecies are passed to the plot command it will plot all
# available s-parameters
ring_slot_sim.plot_s_db(0,0, label='Simulated')
ring_slot_meas.plot_s_db(0,0, label='Measured')
pylab.show()
