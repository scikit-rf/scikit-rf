.. _introduction:

*********************
Introduction to skrf
*********************
.. currentmodule:: skrf.network
.. contents::

This is a brief introduction to **skrf**, highlighting a range of features without going into detail on any single one. At the end of each section are links to other tutorials, that provide more detailed information about a given feature.

This tutotial is aimed at those who have a working python stack, and are somewhat familiar with python. If you are unfamiliar with python, please see scipy's `Getting Started <http://www.scipy.org/Getting_Started>`_ . 

These tutorials are most easily followed by using the ipython_ shell with the ``--pylab`` flag. ::

	> ipython --pylab
	In [1]: 
	
This imports several commonly used functions, and turns on 
`interactive mode <http://matplotlib.org/users/shell.html>`_ , so that plots display immediately. 

.. ipython::
	:suppress:
	
	
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] =120
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()
	

.. note::

	The example code in these tutorials reference files that are distributed with the source package. The working directory for these code snippets is ``scikit-rf/doc/``, hence all data files are referenced relative to that directory. 


For this tutorial, and the rest of the scikit-rf documentation, it is  assumed that **skrf** has been imported as ``rf``. Whether or not you follow this convention in your own code is up to you.


.. ipython::

  In [138]: import skrf as rf

If this produces an error, please see :doc:`installation`. 


Networks
------------------

**skrf** provides an object for a N-port microwave :class:`Network`. A :class:`Network` can be created in a number of ways. One way is from data stored in a touchstone file.

.. ipython::
			
	In [139]: ring_slot = rf.Network('../skrf/data/ring slot.s2p')
 
	
A short description of the network will be printed out if entered onto the command line
	
.. ipython::
	
	In [1]: ring_slot



The :class:`Network` object has numerous other properties and methods which can found in the :class:`Network` docstring. If you are using IPython, then these properties and methods can be 'tabbed' out on the command line. 


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


Additional functionality provided by the :class:`Network` object, such as interpolation, stitching, n-port connections, IO support, and more are described in the  :doc:`networks` tutorial.


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

The :class:`~NetworkSet` object
represents an unordered  set of networks and provides  methods for 
calculating statistical quantities and displaying uncertainty bounds.

A :class:`~NetworkSet` is created from a list or dict of 
:class:`~skrf.network.Network`'s. So first we need to load all of the 
touchstone files. This can be done quickly with 
:func:`~skrf.io.general.read_all` , which loads all skrf-readable objects
in a directory. The argument ``contains`` is used to load only files 
which match a given substring. 


.. ipython::

    In [24]: rf.read_all('../skrf/data/', contains='ro')

This can be passed directly to the :class:`NetworkSet` constructor, 

.. ipython::

    In [24]: ro_dict = rf.read_all('../skrf/data/', contains='ro')
    
    In [24]: ro_ns = rf.NetworkSet(ro_dict, name='ro set') #name is optional
    
    In [24]: ro_ns

A NetworkSet can also be constructed from zipfile of touchstones
through the class method :func:`NetworkSet.from_zip`


Statistical Properties
=======================

Statistical quantities can be calculated by accessing 
properties of the NetworkSet. For example, to calculate the complex 
average of the set, access the ``mean_s`` property


.. ipython::
    
    In [24]: ro_ns.mean_s
    
These methods return a :class:`~skrf.network.Network` object, so they can be 
saved or plotted in the same way as you would with a Network.
To plot the log-magnitude of the complex mean response 

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_mean_s_plot_s_db.png
    In [24]: ro_ns.mean_s.plot_s_db(label='ro')

Or to plot the standard deviation of the complex s-parameters,

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_std_s_plot_s_re.png
    In [24]: ro_ns.std_s.plot_s_re(y_label='Standard Deviations')




Plotting Uncertainty Bounds
===============================

Uncertainty bounds can be plotted through the methods 


.. ipython::
    
    In [24]: figure();
    
    @savefig ns_plot_uncertainty_bounds_s_db.png
    In [24]: ro_ns.plot_uncertainty_bounds_s_db()
    

See the :doc:`networkset` tutorial for more information.
	


.. _ipython: http://ipython.scipy.org/moin/	
