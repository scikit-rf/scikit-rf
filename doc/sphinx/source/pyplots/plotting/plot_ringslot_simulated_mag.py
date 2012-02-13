import pylab
import skrf as rf

# from the extension you know this is a 2-port network
ring_slot= rf.Network('ring slot array simulation.s2p')

pylab.figure(1)
pylab.title('WR-10 Ringslot Array Simulated, Mag')
# if no indecies are passed to the plot command it will plot all
# available s-parameters
ring_slot.plot_s_db()
pylab.show()
