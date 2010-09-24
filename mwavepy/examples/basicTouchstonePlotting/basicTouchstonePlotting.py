'''

'''
import sys
sys.path.append('../')
import mwavepy1 as mvy
import pylab


# load a touchstone file into a 'ntwk' type
horn = mvy.Network('horn.s2p')

# plot the data in some different formats
pylab.figure(1)
pylab.title('Return Loss (Mag)')	
horn.plot_s_db(m=0,n=0)	# m,n are S-Matrix indecies


pylab.figure(2)
pylab.title('Return Loss (Phase)')
# all keyword arguments are passed to matplotlib.plot command
horn.plot_s_deg(0,0, label='Broadband Horn Antenna', color='r', linewidth=2)


pylab.figure(3)
pylab.title('Return Loss (Unwrapped Phase)')
horn.plot_s_deg_unwrapped(0,0)


pylab.figure(4)
horn.plot_s_polar(0,0, show_legend=False)
pylab.title('Return Loss')

# uncomment to save all figures, 
mvy.save_all_figs('.', format = ['png'])
pylab.show()
