.. _introduction:

*********************
Introduction
*********************
.. currentmodule:: skrf.network
.. contents::


Introduction
------------------

This is a brief introduction to **skrf** which highlights a range of features without going into detail on any single one. At the end of each section there are links to other tutorials, that provide more information about a given feature. The intended audience are those who have a working python stack, and are somewhat familiar with python. If you are unfamiliar with python, please see scipy's `Getting Started <http://www.scipy.org/Getting_Started>`_ . 

Although not essential, these tutorials are most easily followed by using the ipython_ shell with the ``--pylab`` flag. ::

	> ipython --pylab
	In [1]: 
	
Using ipython with the ``pylab`` flag imports several commonly used functions, and turns on 
`interactive plotting mode <http://matplotlib.org/users/shell.html>`_  which causes  plots to display immediately. 

.. ipython::
	:suppress:
	
	
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] =120
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()
	



Throughout this tutorial, and the rest of the scikit-rf documentation, it is  assumed that **skrf** has been imported as ``rf``. Whether or not you follow this convention in your own code is up to you.


.. ipython::

  In [138]: import skrf as rf

If this produces an error, please see :doc:`installation`. 

.. note::

	The example code in these tutorials make use of files that are distributed with the source package. The working directory for these code snippets is ``scikit-rf/doc/``, hence all data files are referenced relative to that directory. If you do not have the source package, then you may access these files through the ``skrf.data`` module (ie ``from skrf.data import ring_slot``)


Networks
------------------

**skrf** provides an object for a N-port microwave :class:`Network`. A :class:`Network` can be created in a number of ways. One way is from data stored in a touchstone file.

.. ipython::
			
	In [139]: ring_slot = rf.Network('../skrf/data/ring slot.s2p')
 
	
A short description of the network will be printed out if entered onto the command line
	
.. ipython::
	
	In [1]: ring_slot


he basic attributes of a microwave :class:`Network` are provided by the 
following properties :

* :attr:`Network.s` : Scattering Parameter matrix. 
* :attr:`Network.z0`  : Port Characterisic Impedance matrix.
* :attr:`Network.frequency`  : Frequency Object. 

All of the network parameters are complex :class:`numpy.ndarray` 's of shape *FxNxN*, where *F* is the number of frequency points and *N* is the number of ports. The :class:`Network` object has numerous other properties and methods which can found in the :class:`Network` docstring. If you are using IPython, then these properties and methods can be 'tabbed' out on the command line. 


.. ipython::
	
	@verbatim
	In [1]: short.s<TAB>
	rf.data.line.s              rf.data.line.s_arcl         rf.data.line.s_im
	rf.data.line.s11            rf.data.line.s_arcl_unwrap  rf.data.line.s_mag
	...


Linear Operations 
=========================
	
Element-wise mathematical operations on the scattering parameter matrices are accessible through overloaded operators. To illustrate their usage, load a couple  Networks stored in the ``data`` module. 

.. ipython::
	
	In [21]: short = rf.data.wr2p2_short

	In [22]: delayshort = rf.data.wr2p2_delayshort

	In [22]: short - delayshort

	In [22]: short + delayshort



Cascading and De-embedding
==================================================
Cascading and de-embeding 2-port Networks can also be done though operators. The :func:`cascade` function can be called through the power operator,  ``**``. To calculate a new network which is the cascaded connection of the two individual Networks ``line`` and ``short``, 

.. ipython::
	
	In [21]: short = rf.data.wr2p2_short

	In [22]: line = rf.data.wr2p2_line
	
	In [22]: delayshort = line ** short

De-embedding  can be accomplished by cascading the *inverse* of a network. The inverse of a network is accessed through the property :attr:`Network.inv`. To de-embed the ``short`` from ``delay_short``,

.. ipython::
	
	In [21]: short = line.inv ** delayshort


For more information on the functionality provided by the :class:`Network` object, such as interpolation, stitching, n-port connections, and IO support see the   :doc:`networks` tutorial.


Plotting 
------------
Amongst other things, the methods of the :class:`Network` class provide convenient ways to plot components of the network parameters, 

* :func:`Network.plot_s_db` : plot magnitude of s-parameters in log scale
* :func:`Network.plot_s_deg` : plot phase of s-parameters in degrees
* :func:`Network.plot_s_smith` : plot complex s-parameters on Smith Chart
* ...

To plot all four s-parameters of the ``ring_slot`` on the Smith Chart.

.. ipython::

   @savefig ring_slot,smith.png 
   In [151]: ring_slot.plot_s_smith();


For more detailed information about plotting see the  :doc:`plotting` tutorial



NetworkSet 
--------------

The :class:`~skrf.networkset.NetworkSet` object
represents an unordered  set of networks and provides  methods for 
calculating statistical quantities and displaying uncertainty bounds.

A :class:`~skrf.networkset.NetworkSet` is created from a list or dict of 
:class:`~skrf.network.Network`'s.  This can be done quickly with 
:func:`~skrf.io.general.read_all` , which loads all skrf-readable objects
in a directory. The argument ``contains`` is used to load only files 
which match a given substring. 


.. ipython::

    In [24]: rf.read_all('../skrf/data/', contains='ro')

This can be passed directly to the :class:`~skrf.networkset.NetworkSet` constructor, 

.. ipython::

    In [24]: ro_dict = rf.read_all('../skrf/data/', contains='ro')
    
    In [24]: ro_ns = rf.NetworkSet(ro_dict, name='ro set') #name is optional
    
    In [24]: ro_ns




Statistical Properties
=======================

Statistical quantities can be calculated by accessing 
properties of the NetworkSet. For example, to calculate the complex 
average of the set, access the ``mean_s`` property


.. ipython::
    
    In [24]: ro_ns.mean_s

Similarly, to calculate the complex standard deviation of the set, 

.. ipython::
    
    In [24]: ro_ns.std_s

These methods return a :class:`~skrf.network.Network` object, so the results can be 
saved or plotted in the same way as you would with a Network.
To plot the magnitude of the standard deviation of the set,  

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_std_s_plot_s_re.png
    In [24]: ro_ns.std_s.plot_s_re(y_label='Standard Deviations')



Plotting Uncertainty Bounds
===============================

Uncertainty bounds on any network parameter can be plotted through the methods 


.. ipython::
    
    In [24]: figure();
    
    @savefig ns_plot_uncertainty_bounds_s_db.png
    In [24]: ro_ns.plot_uncertainty_bounds_s_db()
    

See the :doc:`networkset` tutorial for more information.


	
Calibration
----------------------------
.. currentmodule:: skrf.calibration.calibration

**skrf** has support for one and two-port calibration. **skrf**'s\ default calibration algorithms are generic in that they will work with any set of standards. If you supply more calibration standards than is needed, skrf will implement a simple least-squares solution. **skrf** does not currently support TRL.

Calibrations are performed through a :class:`Calibration` class. Creating Networks
a :class:`Calibration` object requires at least two pieces of information:

*   a list of measured :class:`~skrf.network.Network`'s
*   a list of ideal :class:`~skrf.network.Network`'s

The :class:`~skrf.network.Network` elements in each list must all be similar (same #ports, frequency info, etc) and must be aligned to each other, meaning the first element of ideals list must correspond to the first element of measured list.

Optionally, other information can be provided when relevent such as,

*    calibration algorithm
*    enforce eciprocity of embedding networks
*    etc 

When this information is not provided skrf will determine it through 
inspection, or use a default value.

Below is an example script illustrating how to create a :class:`Calibration` .
See the :doc:`calibration` tutorial for more details and examples. 

One Port Calibration
=======================
This example is the same as the first except more concise. ::

    import skrf as rf
    
    my_ideals = rf.load_all_touchstones_in_dir('ideals/')
    my_measured = rf.load_all_touchstones_in_dir('measured/')
    
    
    ## create a Calibration instance
    cal = rf.Calibration(\
	    ideals = [my_ideals[k] for k in ['short','open','load']],
	    measured = [my_measured[k] for k in ['short','open','load']],
	    )
    
.. _ipython: http://ipython.scipy.org/moin/	
