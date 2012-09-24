.. _plotting:

*********
Plotting  
*********
.. currentmodule:: skrf.network
.. contents::


This tutorial illustrates how to create common plots associated with microwave networks. The plotting functions are implemented as methods of the :class:`Network` class, which is provided by the :mod:`skrf.network` module. Below is a list of the some of the plotting functions of network s-parameters, 

   
* :func:`Network.plot_s_re`
* :func:`Network.plot_s_im`
* :func:`Network.plot_s_mag`
* :func:`Network.plot_s_db`
* :func:`Network.plot_s_deg`
* :func:`Network.plot_s_deg_unwrapped`
* :func:`Network.plot_s_rad`
* :func:`Network.plot_s_rad_unwrapped`
* :func:`Network.plot_s_smith`
* :func:`Network.plot_s_complex`



Magnitude
-----------------------

.. plot:: ../pyplots/plotting/plot_ringslot_mag.py
   :include-source:

Phase
---------------------
 
.. plot:: ../pyplots/plotting/plot_ringslot_phase.py
   :include-source:
   
Smith Chart
---------------------
.. plot:: ../pyplots/plotting/plot_ringslot_smith.py
   :include-source:

The Smith Chart can be also drawn independently of any :class:`Network` object through the :func:`~skrf.plotting.smith` function. This function also allows admittance contours can also be drawn.

Multiple S-parameters
----------------------
.. plot:: ../pyplots/plotting/plot_ringslot_simulated_mag.py
   :include-source:

Comparing with Simulation
--------------------------
.. plot:: ../pyplots/plotting/plot_ringslot_simulated_vs_measured.py
   :include-source:

Saving Plots
-------------
Plots can be saved in various file formats using the GUI provided by the matplotlib. However, skrf provides a convenience function, called :func:`~skrf.convenience.save_all_figs`,  that allows all open figures to be saved to disk in multiple file formats, with filenames pulled from each figure's title::

    >>> rf.save_all_figs('.', format=['eps','pdf'])
    ./WR-10 Ringslot Array Simulated vs Measured.eps
    ./WR-10 Ringslot Array Simulated vs Measured.pdf


	
	
