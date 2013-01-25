.. _media:

Media
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

**skrf** supports the  microwave network synthesis based on transmission line models. Network creation is accomplished through methods of the Media class, which represents a transmission line object for a given medium. Once constructed, a :class:`~media.Media` object contains the neccesary properties such as ``propagation constant`` and ``characteristic impedance``, that are needed to generate microwave circuits.

This tutorial illustrates how created Networks using several different :class:`~media.Media` objects. The basic usage is, 

.. ipython:: 

	In [144]: import skrf as rf

	In [144]: freq = rf.Frequency(75,110,101,'ghz')
	
	In [144]: cpw = rf.media.CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
	
	In [144]: cpw.line(100*1e-6, name = '100um line')

More detailed examples illustrating how to create various kinds of Media 
objects are given below. 


.. warning::

	The network creation and connection syntax of **skrf** are cumbersome 
	if you need to doing complex circuit design. For a this type of 
	application, you may be interested in using QUCS_ instead.
	**skrf**'s synthesis cabilities lend themselves more to scripted applications
	such as  `Design Optimization`_ or batch processing.

	

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
	
	In [144]: wg = rf.media.RectangularWaveguide(freq, a=100*rf.mil,z0=50) # see note below about z0
	
	In [144]: wg

See :class:`~rectangularWaveguide.RectangularWaveguide` for details. 

.. note:: 

	The ``z0`` argument in the Rectangular Waveguide constructor is used
	to force a specifc port impedance. This is commonly used to match 
	the port impedance to what a VNA stores in a touchstone file. See 
	:func:`media.Media.__init__` for more information. 
	


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

As an example, plot the cpw's propagation constant vs frequency.

.. ipython:: 
	
	In [144]: plot(cpw.frequency.f_scaled, cpw.propagation_constant.imag);
	
	In [144]: xlabel('Frequency [GHz]');
	
	@savefig media-cpw_propagation_constant.png
	In [144]: ylabel('Propagation Constant [rad/m]');


Because the wave quantities are dynamic they change when the attributes 
of the cpw line change. To illustrate this, plot the propagation constant of the cpw for various values of substrated permativity,  

.. ipython:: 
	
	In [144]: figure();

	In [47]: for ep_r in [9,10,11]:
	   ....:     cpw.ep_r = ep_r
	   ....:     plot(cpw.frequency.f_scaled, cpw.propagation_constant.imag, label='er=%.1f'%ep_r)
	
	In [144]: xlabel('Frequency [GHz]');
	
	In [144]: ylabel('Propagation Constant [rad/m]');
	
	@savefig media-cpw_propagation_constant2.png
	In [144]: legend();
	
	@supress
	In [144]: cpw.ep_r = 10.6
	
Network Synthesis
--------------------

Networks are created through methods of a Media object.  Here is a brief
list of some generic network components skrf supports,

* :func:`~media.Media.match`
* :func:`~media.Media.short`
* :func:`~media.Media.open`
* :func:`~media.Media.load`
* :func:`~media.Media.line`
* :func:`~media.Media.thru`
* :func:`~media.Media.tee`
* :func:`~media.Media.delay_short`
* :func:`~media.Media.shunt_delay_open`

Usage of these methods can is demonstrated below.

To create a 1-port network for a rectangular waveguide short, 

.. ipython:: 

	In [144]: wg.short(name = 'short') 

Or to create a :math:`90^{\circ}` section of cpw line, 

.. ipython:: 

	In [144]: cpw.line(d=90,unit='deg', name='line')

.. note::

	Simple circuits like :func:`~media.Media.short` 
	and :func:`~media.Media.open` are ideal short and opens with
	:math:`\Gamma = -1` and :math:`\Gamma = 1`, i.e. they dont take 
	into account sophisticated effects of the discontinuties.
	Effects of discontinuities are implemented as methods specific to a 
	given Media, like :func:`CPW.cpw_short <cpw.CPW.cpw_short>`.
	

Building Cicuits
----------------------

By connecting a series of simple circuits, more complex circuits can be 
made. To build a the :math:`90^{\circ}` delay short, in the 
rectangular waveguide media defined above.

.. ipython:: 

	In [144]: delay_short = wg.line(d=90,unit='deg') ** wg.short()
	
	In [144]: delay_short.name = 'delay short'
	
	In [144]: delay_short

When Networks with more than 2 ports need to be connected together, use 
:func:`rf.connect() <skrf.network.connect>`.  To create a two-port network for a shunted delayed open, you can create an ideal 3-way splitter (a 'tee') and conect the delayed open to one of its ports,
	
.. ipython:: 

	In [14]: tee = cpw.tee()
	
	In [14]: delay_open = cpw.delay_open(40,'deg')
	
	In [14]: shunt_open = rf.connect(tee,1,delay_open,0)


If a specific circuit is created frequenctly, it may make sense to 
use a function to create the circuit. This can be done most quickly using lamba

.. ipython:: 

	In [144]: delay_short = lambda d: wg.line(d,'deg')**wg.short()
	
	In [144]: delay_short(90)
	
This is how many of **skrf**'s network creation methods are made internally.

A more useful example may be to create a function for a shunt-stub tuner,
that will work for any media object

.. ipython:: 

	In [14]: def shunt_stub(med, d0, d1):
	   ....:     return med.line(d0,'deg')**med.shunt_delay_open(d1,'deg')
	
	In [14]: shunt_stub(cpw,10,90)





Design Optimization
-------------------

The abilities of scipy_'s optimizers can be used to automate network design. In this example, skrf is used to automate the single stub design. First, we create a 'cost' function which returns somthing we want to minimize, such as the reflection coefficient magnitude at band center. Then, one of scipy's minimization algorithms is used to determine the optimal parameters of the stub lengths to minimize this cost.

.. ipython:: 

	In [14]: from scipy.optimize import fmin
	
	# the load we are trying to match
	In [14]: load = cpw.load(rf.zl_2_Gamma0(z0=50,zl=100))
	
	# single stub circuit generator function
	In [14]: def shunt_stub(med, d0, d1):
	   ....:     return med.line(d0,'deg')**med.shunt_delay_open(d1,'deg')
	
	
	# define the cost function we want to minimize (this uses sloppy namespace)
	In [14]: def cost(d):
	   ....:     return (shunt_stub(cpw,d[0],d[1]) ** load)[100].s_mag.squeeze()
	
	# initial guess of optimal delay lengths in degrees
	In [14]: d0 = 120,40 # initial guess
	
	#determine the optimal delays
	In [14]: d_opt = fmin(cost,(120,40))
	
	In [14]: d_opt 

References
--------------

.. [#] http://www.microwaves101.com/encyclopedia/coplanarwaveguide.cfm

.. _scipy: http://www.scipy.org

.. _QUCS: http://www.qucs.sourceforge.net
