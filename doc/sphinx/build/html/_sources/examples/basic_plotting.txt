.. _example-basic-plotting:
****************************
Basic Plotting  
****************************

This example illustrates how to create common plots::

	import mwavepy as mv
	import pylab
	
	
	# create a Network type from a touchstone file of a horn antenna
	horn = mv.Network('horn.s1p')
	
	# plot magnitude of S11
	pylab.figure(1)
	pylab.title('Return Loss (Mag)')	
	horn.plot_s_db(m=0,n=0)	# m,n are S-Matrix indecies
	# show the plots (only needed if you dont have interactive set on ipython)
	pylab.show()

.. figure::  ../images/Return_Loss(Mag).png
   :align:   center
   :width:	800

::	

	# plot phase of S11
	pylab.figure(2)
	pylab.title('Return Loss (Phase)')
	# all keyword arguments are passed to matplotlib.plot command
	horn.plot_s_deg(0,0, label='Broadband Horn Antenna', color='r', linewidth=2)

.. figure::  ../images/Return_Loss(Phase).png
   :align:   center
   :width:	800

::	
	
	# plot unwrapped phase of S11
	pylab.figure(3)
	pylab.title('Return Loss (Unwrapped Phase)')
	horn.plot_s_deg_unwrapped(0,0)
	
.. figure::  ../images/Return_Loss(Unwrapped_Phase).png
   :align:   center
   :width:	800

::	

	# plot complex S11 on smith chart
	pylab.figure(5)
	horn.plot_s_smith(0,0, show_legend=False)
	pylab.title('Return Loss, Smith')

.. figure::  ../images/Return_Loss(Smith).png
   :align:   center
   :width:	800

::		

	# plot complex S11 on polar grid
	pylab.figure(4)
	horn.plot_s_polar(0,0, show_legend=False)
	pylab.title('Return Loss, Polar')

.. figure::  ../images/Return_Loss(Polar).png
   :align:   center
   :width:	800

::	



	#  to save all figures, 
	mv.save_all_figs('.', format = ['png','eps'])
	
	
	
