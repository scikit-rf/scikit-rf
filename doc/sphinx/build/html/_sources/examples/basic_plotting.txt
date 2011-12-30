.. _example-basic-plotting:
****************************
Basic Plotting  
****************************

.. currentmodule:: mwavepy.network

This example illustrates how to create common plots. The plotting 
functions are implemented as methods of the :class:`Network` class, 
which is provided by the :mod:`mwavepy.network` module. 
Below is a brief list of the some commonly used plotting functions, 

   
   Network.plot_s_smith
   Network.plot_s_complex
   Network.plot_s_db 
   Network.plot_s_mag 
   Network.plot_s_deg 
   Network.plot_s_deg_unwrapped 
   Network.plot_s_rad 
   Network.plot_s_rad_unwrapped 
   Network.plot_s_quad 
   Network.plot_s_quad_unwrapped 
   Network.plot_s_re 
   Network.plot_s_im 

Return Loss Magnitude
-----------------------

.. plot:: ../pyplots/basic_plotting/plot_ringslot_mag.py
   :include-source:

Return Loss Phase
---------------------
 
.. plot:: ../pyplots/basic_plotting/plot_ringslot_phase.py
   :include-source:
   
Return Loss Smith
---------------------
 
.. plot:: ../pyplots/basic_plotting/plot_ringslot_smith.py
   :include-source:

All S-parameters
---------------------
.. plot:: ../pyplots/basic_plotting/plot_ringslot_simulated_mag.py
   :include-source:

Comparing with Simulation
--------------------------
.. plot:: ../pyplots/basic_plotting/plot_ringslot_simulated_vs_measured.py
   :include-source:

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
	
	
	
