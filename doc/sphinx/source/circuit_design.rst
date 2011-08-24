.. _circuit-design:

Circuit Design
**********************

Intro
----------

mwavepy has basic support for microwave circuit design. Network synthesis is accomplished through the  (awkwardly named) :doc:`auto_workingband` Class. A :doc:`auto_workingband` has three main properties:

* transmission line (object)
* frequency (object)
* z0 (array)

The transmission line object contains properties inherent to the medium, such as propagation constant and characteristic impedance, which are needed to generate certain networks. See the :doc:`transmission_line` page for more details about transmission line objects.

Most circuit design is done within a given frequency band. The frequency object is used to relive the user of repitously providing frequency information for each new network created. 

z0 stores the characteristic impedance of the medium, which may be frequency dependent. If this is not explicitly supplied, mwavepy will attempt to determine it through inspection of the tranmission line object.


Initializing a WorkingBand
------------------------------

Here is an example of how to initialize a :doc:`auto_workingband` class representing freespace from 10-20GHz::

	import mwavepy as mv
	tline = mv.transmissionLine.FreeSpace()
	frequency = mv.Frequency(10,20,101,'ghz')
	z0= 377
	
	wb = mv.WorkingBand(tline=tline, frequency=frequency, z0=z0)

Although all arguments were explicitly passed to the constructor in this example, this doesnt have to be the case. If the characteristic impedance is omitted, the constructor will try and generate it by accessing the Z0() function of the transmission line object. Continuing from the previous code, the WorkingBand could be constructed without a z0::

	wb = mv.WorkingBand(tline=tline, frequency=frequency)
	print wb.z0
	
	[ 376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j
	376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j
	376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j




As a more complex example, if the transmission line class has frequency information associated with it (which is neccesary for non-TEM cases), then :doc:`auto_workingband` will attempt to generate the other information through inspection. For example::
	
	gamma = linspace(5000,1000,101)
	z0 = linspace(50,75,101)
	f = linspace(10,20,101)*1e9
	# create a non-TEM transmission line
	tline = mv.transmissionLine.GenericTEM.from_gamma_z0(gamma,z0,f)
	
	wb = mv.WorkingBand(tline)


Creating Individual Networks
------------------------------

Network components are created through methods of :doc:`auto_workingband` object. Here is a short, imcomplete list of a some network components mwavepy supports,

* match
* short
* open 
* load
* line
* tee
* thru
* delay_short
* shunt_delay_open

Details for each component and usage help can be found in their doc-strings. So help(wb.short) should provide you with enough details to create a short-circuit component. 
To create a 1-port network for a short, ::

	wb.short() 

to create a 90deg section of transmission line, with characteristic impedance of 30 ohms::

	wb.line(d=90,unit='deg',z0=30)
	
 

 

Building Cicuits
----------------------

Circuits can be built in an intuitive maner from individual networks. To build a the 90deg delay_short standard can be made by::

	delay_short_90deg = wb.line(90,'deg') ** wb.short()


For frequently used circuits, it may be worthwhile creating a function for something like this::

	def delay_short(wb,*args,**kwargs):
		return wb.line(*args,**kwargs)**wb.short()
	
	delay_short(wb,90,'deg')

This is how many of mwavepy's network compnents are made internally. 

To connect networks with more than two ports together, use the *connect()* function. You must provide the connect function with the two networks to be connected and the port indecies (starting from 0) to be connected. 

To connect port# '0' of ntwkA to port# '3' of ntwkB: ::
	
	ntwkC = mv.connect(ntwkA,0,ntwkB,3)

Note that the connect function takes into account port impedances. To create a two-port network for a shunted delayed open, you can create an ideal 3-way splitter (a 'tee') and conect the delayed open to one of its ports, like so::

	tee = wb.tee()
	delay_open = wb.delay_open(40,'deg')
	
	shunt_open = connect(tee,1,delay_open,0)


Single Stub Tuner
--------------------

This is an example of how to design a single stub tuning network to match a 100ohm resistor to a 50 ohm environment. ::
	
	# calculate reflection coefficient off a 100ohm
	Gamma0 = mv.zl_2_Gamma0(z0=50,zl=100)	
	
	# create the network for the 100ohm load
	load = wb.load(Gamma0)
	
	# create the single stub  network, parameterized by two delay lengths
	# in units of 'deg'
	single_stub = wb.shunt_delay_open(120,'deg') ** wb.line(40,'deg')
	
	# the resulting network
	result = single_stub ** load 
	
	result.plot_s_db()


Optimizing Designs
-------------------
The abilities of scipy's optimizers can be used to automate network design. To automate the single stub design, we can create a 'cost' function which returns somthing we want to minimize, such as the reflection coefficient magnitude at band center.  
::

	from scipy.optmize import fmin
	
	# the load we are trying to match
	load = wb.load(mv.zl_2_Gamma0(100))
	
	# single stub generator function
	def single_stub(wb,d0,d1):
		return wb.shunt_open(d1,'deg')**wb.line(d0,'deg')
	
	# cost function we want to minimize (note: this uses sloppy namespace)
	def cost(d):
		return (single_stub(wb,d[0],d[1]) ** load)[100].s_mag.squeeze()
	
	
	# initial guess of optimal delay lengths in degrees
	d0= 120,40 # initial guess
	
	#determine the optimal delays
	d_opt = fmin(cost,(120,40))
	


