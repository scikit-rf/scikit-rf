.. _circuit-synthesis:

Circuit Synthesis
**********************

.. currentModule:: skrf.media

.. ipython::
	:suppress:
	
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] =120
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()
	
.. contents::

Introduction
-------------

**skrf** supports the  microwave network synthesis based on transmission line models. Network creation is accomplished through methods of the Media class (see :mod:`skrf.media`), which represents a transmission line object for a given medium. Once constructed, a :class:`~media.Media` object contains the neccesary properties such as ``propagation constant`` and ``characteristic impedance``, that are needed to generate microwave circuits.

This tutorial illustrates how created Networks using several different :class:`~media.Media` objects. The basic usage is, 

.. ipython:: 

	In [144]: import skrf as rf

	In [144]: freq = rf.Frequency(75,110,101,'ghz')
	
	In [144]: cpw = rf.media.CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
	
	In [144]: cpw.line(100*1e-6, name = '100um line')

More detailed examples illustrating how to create various kinds of Media 
objects are given below. 


Media's Supported by skrf
==========================

The base-class, :class:`~media.Media`,  is constructed directly from 
values of propagation constant and characteristic impedance. Specific 
instances of Media objects can be created from relevant physical and 
electrical properties. Below is a list of mediums types supported by skrf,

* :class:`~cpw.CPW`
* :class:`~rectangularWaveguide.RectangularWaveguide`
* :class:`~freespace.Freespace`
* :class:`~distributedCircuit.DistributedCircuit`
* :class:`~media.Media`


Creating :class:`~media.Media` Objects
---------------------------------------------

Typically, network analysis is done within a given frequency band. When a :class:`~media.Media` object is created, it must be given  a  :class:`~skrf.frequency.Frequency` object. This prevent having to repitously provide frequency information for each new network created. 

Coplanar Waveguide
====================

Here is an example of how to initialize a coplanar waveguide [#]_ media. The instance has  a 10um center conductor, gap of 5um, and substrate with relative permativity of 10.6,

.. ipython:: 

	In [144]: import skrf as rf

	In [144]: freq = rf.Frequency(75,110,101,'ghz')
	
	In [144]: cpw = rf.media.CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
	
	In [144]: cpw
	


See :class:`~cpw.CPW` for details on that class.


Freespace
==============

Here is another example, this time constructing a plane-wave in freespace from 10-20GHz 

.. ipython:: 
	
	In [144]: freq = rf.Frequency(10,20,101,'ghz')
	
	In [144]: fs = rf.media.Freespace(freq)
	
	In [144]: fs

See :class:`~freespace.Freespace` for details.


Rectangular Waveguide
=====================

or a WR-10 Rectangular Waveguide

.. ipython:: 

	In [144]: freq = rf.Frequency(75,110,101,'ghz')
	
	In [144]: wg = rf.media.RectangularWaveguide(freq, a=100*rf.mil)
	
	In [144]: wg

See :class:`~rectangularWaveguide.RectangularWaveguide` for details.


Working with Media's
---------------------

Once constructed, the pertinent wave quantities of the media such as 
propagation constant and characteristic impedance can be accessed through
the properties :attr:`~media.Media.propagation_constant` and 
:attr:`~media.Media.characteristic_impedance`. These properties return 
complex :class:`numpy.ndarray`'s, 

.. ipython:: 
	
	In [144]: cpw.propagation_constant[:3]

	In [144]: cpw.characteristic_impedance[:3]

As an example, we plot 
these properties for the cpw class defined above.

.. ipython:: 
	
	In [144]: plot(cpw.frequency.f_scaled, cpw.propagation_constant.imag);
	
	In [144]: xlabel('Frequency [GHz]');
	
	@savefig circuit_synthesis-cpw_propagation_constant.png
	In [144]: ylabel('Propagation Constant [rad/m]');



Once created, the attributes of the cpw line can be changed  and the 
propagation constant and impedance change appropriatly. To illustrate, 
we plot the propagation constant of the cpw for various 


Network Synthesis
--------------------

Network components are created through methods of a Media object.  Here is a brieflist of some generic network components skrf supports,

* :func:`~media.Media.match`
* :func:`~media.Media.short`
* :func:`~media.Media.open`
* :func:`~media.Media.load`
* :func:`~media.Media.line`
* :func:`~media.Media.thru`
* :func:`~media.Media.tee`
* :func:`~media.Media.delay_short`
* :func:`~media.Media.shunt_delay_open`


To create a 1-port network for a rectangular waveguide short, 

.. ipython:: 

	In [144]: wg.short() 

.. note::
	Simple circuits like :func:`~media.Media.short` 
	and :func:`~media.Media.open` are ideal short and opens, meaning 
	they have :math:`\Gamma = -1` and :math:`\Gamma = 1`, i.e. they dont take 
	into account sophisticated effects of the discontinuties.
	Effects of discontinuities are implemented as methods specific to a 
	given Media class, like :func:`CPW.cpw_short <cpw.CPW.cpw_short>`.
	


Create a :math:`90^{\circ}` section of transmission line, with characteristic impedance of 30 :math:`\Omega`

.. ipython:: 

	In [144]: cpw.line(d=90,unit='deg',z0=30)


Network components specific to a given medium, such as cpw_short, or microstrip_bend, are implemented in by the Media Classes themselves.

 

Building Cicuits
----------------------

Circuits can be built in an intuitive maner from individual networks. To build a the 90deg delay_short standard can be made by::

	delay_short_90deg = my_media.line(90,'deg') ** my_media.short()


For frequently used circuits, it may be worthwhile creating a function for something like this::

	def delay_short(wb,*args,**kwargs):
		return my_media.line(*args,**kwargs)**my_media.short()
	
	delay_short(wb,90,'deg')

This is how many of skrf's network compnents are made internally. 

To connect networks with more than two ports together, use the *connect()* function. You must provide the connect function with the two networks to be connected and the port indecies (starting from 0) to be connected. 

To connect port# '0' of ntwkA to port# '3' of ntwkB: ::
	
	ntwkC = rf.connect(ntwkA,0,ntwkB,3)

Note that the connect function takes into account port impedances. To create a two-port network for a shunted delayed open, you can create an ideal 3-way splitter (a 'tee') and conect the delayed open to one of its ports, like so::

	tee = my_media.tee()
	delay_open = my_media.delay_open(40,'deg')
	
	shunt_open = connect(tee,1,delay_open,0)


Single Stub Tuner
--------------------

This is an example of how to design a single stub tuning network to match a 100ohm resistor to a 50 ohm environment. ::
	
	# calculate reflection coefficient off a 100ohm
	Gamma0 = rf.zl_2_Gamma0(z0=50,zl=100)	
	
	# create the network for the 100ohm load
	load = my_media.load(Gamma0)
	
	# create the single stub  network, parameterized by two delay lengths
	# in units of 'deg'
	single_stub = my_media.shunt_delay_open(120,'deg') ** my_media.line(40,'deg')
	
	# the resulting network
	result = single_stub ** load 
	
	result.plot_s_db()


Optimizing Designs
-------------------
The abilities of scipy's optimizers can be used to automate network design. To automate the single stub design, we can create a 'cost' function which returns somthing we want to minimize, such as the reflection coefficient magnitude at band center.  
::

	from scipy.optmize import fmin
	
	# the load we are trying to match
	load = my_media.load(rf.zl_2_Gamma0(100))
	
	# single stub generator function
	def single_stub(wb,d0,d1):
		return my_media.shunt_open(d1,'deg')**my_media.line(d0,'deg')
	
	# cost function we want to minimize (note: this uses sloppy namespace)
	def cost(d):
		return (single_stub(wb,d[0],d[1]) ** load)[100].s_mag.squeeze()
	
	
	# initial guess of optimal delay lengths in degrees
	d0= 120,40 # initial guess
	
	#determine the optimal delays
	d_opt = fmin(cost,(120,40))
	


References
--------------

.. [#] http://www.microwaves101.com/encyclopedia/coplanarwaveguide.cfm
