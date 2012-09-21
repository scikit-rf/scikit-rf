.. _circuit-design:

Circuit Design
**********************
.. contents::
Intro
----------

skrf has basic support for microwave circuit design. Network synthesis is accomplished through the Media Class (:py:mod:`skrf.media`), which represent a transmission line object for a given medium. A  Media object contains properties such as propagation constant and characteristic impedance, that are needed to generate network components.

Typically circuit design is done within a given frequency band. Therefore every Media object is created with a  Frequency object to  relieve the user of repitously providing frequency information for each new network created. 


Media's Supported by skrf
------------------------------

Below is a list of mediums types supported by skrf,

* DistributedCircuit
* Freespace
* RectangularWaveguide
* CPW

More info on all of these classes can be found in the media sub-module section of :py:mod:`skrf.media`  mavepy's API. 

Here is an example of how to initialize a Media object representing a freespace from 10-20GHz::

	import skrf as rf
	freq = rf.Frequency(10,20,101,'ghz')
	my_media = rf.media.Freespace(freq)

Here is another example constructing a coplanar waveguide media. The instance has  a 10um center conductor and gap of 5um, on a substrate with relative permativity of 10.6,::

	freq = rf.Frequency(500,750,101,'ghz')
	my_media = rf.media.CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)

or a WR10 Rectangular Waveguide::
	
	from scipy.constants import * # for the 'mil' unit
	freq = rf.Frequency(75,110,101,'ghz')
	my_media = rf.media.RectangularWaveguide(freq, a=100*mil)
	
Creating Individual Networks
------------------------------

Network components are created through methods of a Media object.  Here is a brief, incomplete list of a some generic network components skrf supports,

* match
* short
* open 
* load
* line
* tee
* thru
* delay_short
* shunt_delay_open

Details for each component and usage help can be found in their doc-strings. So help(my_media.short) should provide you with enough details to create a short-circuit component. 
To create a 1-port network for a short, ::

	my_media.short() 

to create a 90deg section of transmission line, with characteristic impedance of 30 ohms::

	my_media.line(d=90,unit='deg',z0=30)
	
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
	


