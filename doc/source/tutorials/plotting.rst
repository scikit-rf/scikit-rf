.. _plotting:

*********
Plotting  
*********
.. currentmodule:: skrf.network
.. contents::


.. ipython::
	:suppress:
		
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] = 120 
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()
    



Plotting Methods
-----------------------

Network plotting abilities are implemented as methods of the :class:`Network` class. Some of the plotting functions of network s-parameters are, 

.. hlist::
    :columns: 2

    * :func:`Network.plot_s_re`
    * :func:`Network.plot_s_im`
    * :func:`Network.plot_s_mag`
    * :func:`Network.plot_s_db`
    * :func:`Network.plot_s_deg`
    * :func:`Network.plot_s_deg_unwrap`
    * :func:`Network.plot_s_rad`
    * :func:`Network.plot_s_rad_unwrap`
    * :func:`Network.plot_s_smith`
    * :func:`Network.plot_s_complex`

Similar methods exist for Impedance (:attr:`Network.z`) and Admittance Parameters (:attr:`Network.y`), 

.. hlist::
    :columns: 2
    
    * :func:`Network.plot_z_re`
    * :func:`Network.plot_z_im`
    * ...
    * :func:`Network.plot_y_re`
    * :func:`Network.plot_z_im`
    * ...

Step-by-step examples of how to create and customize plots are given 
below. 


Complex Plots
-----------------------

Smith Chart 
+++++++++++++++++

As a first example, load a :class:`~skrf.network.Network` from the
``data`` module, and plot all four  s-parameters on the Smith chart.

.. ipython::

    In [138]: import skrf as rf
	
    In [139]: from skrf.data import ring_slot
    
    In [139]: ring_slot
    
    @savefig plotting-ring_slot,smith.png
    In [139]: ring_slot.plot_s_smith()

.. note:: 

    If you dont see any plots after issuing these commands, then you may not have  started ipython with the ``--pylab`` flag. Try ``from pylab import *`` to import the matplotlib commands and ``ion()``  to turn on interactive plotting. See `this page <http://matplotlib.org/users/shell.html>`_ , for more info on ipython's `pylab` mode.

.. note:: 
    
    Why do my plots look different? See  `Formating Plots`_


The smith chart can be drawn with some impedance values labeled through the ``draw_labels`` argument.

.. ipython::
    
    In [139]: figure();
    
    @savefig plotting-ring_slot,smith1.png
    In [139]: ring_slot.plot_s_smith(draw_labels=True)


Another common option is to draw addmitance contours, instead of impedance. This is controled through the  ``chart_type`` argument.

.. ipython::
    
    In [139]: figure();
    
    @savefig plotting-ring_slot,smith2.png
    In [139]: ring_slot.plot_s_smith(chart_type='y')
    
See :func:`~skrf.plotting.smith` for more info on customizing the Smith Chart. 

.. note:: 

    If more than one ``plot_s_smith()`` command is issued on a single figure, you may need to call :func:`draw()` to refresh the  chart. 


Complex Plane 
+++++++++++++++

Network parameters can also be plotted in the complex plane without a Smith Chart through :func:`Network.plot_s_complex`.

.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,complex.png
    In [138]: ring_slot.plot_s_complex();
    


Rectangular Plots
---------------------

Log-Magnitude 
++++++++++++++++

Scalar components of the complex network parameters can be plotted vs 
frequency as well. To plot the log-magnitude of the s-parameters vs. frequency,

.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,db.png
    In [138]: ring_slot.plot_s_db()
    
When no arguments are passed to the plotting methods, all parameters are plotted. Single parameters can be plotted by passing indecies ``m`` and ``n`` to the plotting commands (indexing start from 0). Comparing the simulated reflection coefficient off the ring slot to a measurement, 

.. ipython::
    
    In [138]: from skrf.data import ring_slot_meas
    
    In [138]: figure();
    
    In [138]: ring_slot.plot_s_db(m=0,n=0) # s11
    
    @savefig plotting-ring_slot,db2.png
    In [138]: ring_slot_meas.plot_s_db(m=0,n=0) # s11

See `Customizing Plots`_ for more information on customization. 


Phase 
++++++++++++++++

Plot phase, 

.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,deg.png
    In [138]: ring_slot.plot_s_deg()
    
Or unwrapped phase, 

.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,deg_unwrap.png
    In [138]: ring_slot.plot_s_deg_unwrap()

Impedance, Admittance 
++++++++++++++++++++++

The components the Impendanc and Admittance parameters can be plotted 
similarly, 

.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,z_im.png
    In [138]: ring_slot.plot_z_im()
    
.. ipython::

    In [138]: figure();
    
    @savefig plotting-ring_slot,y_re.png
    In [138]: ring_slot.plot_y_re()

Customizing Plots
-------------------

The legend entries are automatically filled in with the Network's :attr:`~skrf.network.Network.name`. The entry can be overidden by passing the ``label`` argument to the plot method.
 
.. ipython::
    
    In [138]: figure();
    
    In [138]: ring_slot.plot_s_db(m=0,n=0, label = 'Simulation')
    
    @savefig plotting-ring_slot,db3.png
    In [138]: ring_slot_meas.plot_s_db(m=0,n=0, label = 'Measured')

The frequency unit used on the x-axis is automatically filled in from 
the Networks :attr:`~skrf.network.Network.frequency` attribute. To change
the label, change the frequency's ``unit``.
    
.. ipython::
    
    @verbatim
    In [138]: ring_slot.frequency.unit = 'mhz'


Other key word arguments given to the plotting methods are passed through to the matplotlib :func:`~matplotlib.pyplot.plot` function. 

.. ipython::
    
    In [138]: figure();
    
    In [138]: ring_slot.plot_s_db(m=0,n=0, linewidth = 3, linestyle = '--', label = 'Simulation')
    
    @savefig plotting-ring_slot,db4.png
    In [138]: ring_slot_meas.plot_s_db(m=0,n=0, marker = 'x', markevery = 10,label = 'Measured')
    
All components of the plots can be customized through  `matplotlib <http://matplotlib.sourceforge.net>`_  functions. 


.. ipython::

    In [138]: figure();
    
    In [138]: ring_slot.plot_s_smith()
    
    In [138]: xlabel('Real Part');
    
    In [138]: ylabel('Imaginary Part');
    
    In [138]: title('Smith Chart');
    
    @savefig plotting-ring_slot,smith3.png
    In [139]: draw();    
   


Saving Plots
-------------
Plots can be saved in various file formats using the GUI provided by the matplotlib. However, skrf provides a convenience function, called :func:`~skrf.plotting.save_all_figs`,  that allows all open figures to be saved to disk in multiple file formats, with filenames pulled from each figure's title::

    >>> rf.save_all_figs('.', format=['eps','pdf'])
    ./WR-10 Ringslot Array Simulated vs Measured.eps
    ./WR-10 Ringslot Array Simulated vs Measured.pdf


Misc 
---------------

Adding Markers to Lines
++++++++++++++++++++++++++

A common need is to make a color plot, interpretable in greyscale print. 
There is a convenient function, :func:`~skrf.plotting.add_markers_to_lines`, which  adds markers each line in a plots *after* the plot has been made. In this way, adding markers to an already written set of plotting commands is easy.

	
.. ipython::
    
    In [138]: figure();
    
    In [138]: ring_slot.plot_s_db(m=0,n=0)
    
    In [138]: ring_slot_meas.plot_s_db(m=0,n=0)

	@savefig plotting-ring_slot,db6.png
    In [138]: rf.add_markers_to_lines()


Formating Plots
+++++++++++++++++++

It is likely that your plots dont look exactly like the ones in this 
tutorial. This is because matplotlib supports a vast amount of `customization <http://matplotlib.org/users/customizing.html>`_.  Formating options can be customized `on-the-fly` by modifying values of the ``rcParams`` dictionary. Once these are set to your liking they can be saved to your ``.matplotlibrc`` file. 



Here are some relevant parameters which should get your plots looking close to the ones in this tutorial::
    
    my_params = {
    'figure.dpi':  120,
    'figure.figsize': [4,3],
    'figure.subplot.left' : 0.15,
    'figure.subplot.right'	: 0.9,    
    'figure.subplot.bottom'	: 0.12, 
    'axes.titlesize'    : 'medium',    
    'axes.labelsize'    : 10 ,
    'ytick.labelsize'   :'small',
    'xtick.labelsize'   :'small',
    'legend.fontsize'   : 8  #small,
    'legend.loc'	    : 'best',
    'font.size'         : 10.0,
    'font.family'       : 'serif',
    'text.usetex' : True,    # if you have latex
    }
    
    rcParams.update(my_params) 
    
The project `mpltools <http://tonysyu.github.com/mpltools/>`_ provides a way  to  switch between pre-defined `styles`, and contains other useful plotting-related features. 

