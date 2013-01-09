.. _networkset:

***********************
NetworkSet
***********************
.. currentmodule:: skrf.networkSet
.. contents::


.. ipython::
	:suppress:
	
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] =120
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()

The :class:`~NetworkSet` object
represents an unordered  set of networks and provides  methods for 
calculating statistical quantities and displaying uncertainty bounds.

Creating a :class:`NetworkSet`
-------------------------------

For this example, assume that numerous measurements of a single network 
are made. These measurements have been retrieved from a VNA and are
in the form of touchstone files. A set of example data can be found in 
``scikit-rf/skrf/data/``, with naming convention ``ro,*.s1p``, 

.. ipython::

    In [24]: import skrf as rf
    
    In [24]: ls ../skrf/data/ro*


The files ``ro,1.s1p`` , ``ro,2.s1p``, ...  are redundant measurements on 
which we would like to calculate statistics using the :class:`~NetworkSet`
class.

A :class:`~NetworkSet` is created from a list or dict of 
:class:`~skrf.network.Network`'s. So first we need to load all of the 
touchstone files into Networks. This can be done quickly with 
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

Using a :class:`NetworkSet`
----------------------------

The :class:`~skrf.network.Network` elements in a :class:`NetworkSet` can be accessed like the elements of list, 

.. ipython::
    
    In [24]: ro_ns[0]

Most :class:`~skrf.network.Network` methods are also methods of 
:class:`NetworkSet`. These methods are called on each 
:class:`~skrf.network.Network` element individually. For example to 
plot the log-magnitude of the s-parameters of each Network,  
(see :doc:`plotting` for details on :class:`~skrf.network.Network`
ploting methods).

.. ipython::
    
    @savefig ns_plot_s_db.png
    In [24]: ro_ns.plot_s_db()


Some statistical quantities can be calculated by accessing 
properties of the NetworkSet. For example, to calculate the complex 
average of the set, access the ``mean_s`` property


.. ipython::
    
    In [24]: ro_ns.mean_s
    
The naming convention of the statistical operator properties are `NetworkSet.function_parameter`, where `function` is the name of the 
statistical function, and `parameter` is the Network parameter to operate 
on. These methods return a :class:`Network` object, so they can be 
saved or plotted in the same way as you would with a Network.
To plot the complex mean response on the smith chart

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_mean_s_plot_s_smith.png
    In [24]: ro_ns.mean_s.plot_s_smith(r = .5)

Its possible to calculate statistical quantities on the scalar 
components of the complex network parameters. To calculate the 
mean of the phase component,

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_mean_s_deg.png
    In [24]: ro_ns.mean_s_deg.plot_s_re()
    
    
Uncertainty Bounds
----------------------------

.. ipython::
    
    In [24]: figure();
    
    @savefig ns_plot_uncertainty_bounds_s_db.png
    In [24]: ro_ns.plot_uncertainty_bounds_s_db()
    
    In [24]: figure();
    
    @savefig ns_plot_uncertainty_bounds_s_deg.png
    In [24]: ro_ns.plot_uncertainty_bounds_s_deg()

