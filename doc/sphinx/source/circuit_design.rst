.. _circuit-design:

Circuit Design
**********************

Intro
----------

mwavepy has basic support for microwave circuit design. Network synthesis is accomplished through the  (awkwardly named) :doc:`auto_workingband` Class. A :doc:`auto_workingband` has three main properties:

* transmission line (object)
* frequency (object)
* z0 (array)

The transmission line object contains properties inherent to the medium, such as propagation constant and characteristic impedance, which are needed to generate networks containing transmission lines. 


Please see the :doc:`transmission_line` page for more details about transmission line objects.


Initializing a WorkingBand
------------------------------

An example of how to initialize a :doc:`auto_workingband` class representing freespace from 10-20GHz::

	import mwavepy as mv
	tline = mv.transmissionLine.FreeSpace()
	frequency = mv.Frequency(10,20,101,'ghz')
	z0= 377
	
	wb = mv.WorkingBand(tline=tline, frequency=frequency, z0=z0)

Although all arguments where explicitly passed to the constructor in this example, this doesnt have to be the case. If the characteristic impedance is omitted, the constructor will try and generate it by accessing the Z0() function of the tline object. Continuing from the previous code, the WorkingBand could be constructed without a z0::

	wb = mv.WorkingBand(tline=tline, frequency=frequency)
	print wb.z0
	
	[ 376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j
	376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j
	376.73031346+0.j  376.73031346+0.j  376.73031346+0.j  376.73031346+0.j




For a more complex example, if the transmission line class has frequency information associated with it (which is neccesary for non-TEM cases), then :doc:`auto_workingband` will attempt to generate the other information through inspection. For example::
	
	gamma = linspace(5000,1000,101)
	z0 = linspace(50,75,101)
	f = linspace(10,20,101)*1e9
	# create a non-TEM transmission line
	tline = mv.transmissionLine.GenericTEM.from_gamma_z0(gamma,z0,f)
	
	wb = mv.WorkingBand(tline)


Creating Networks
--------------------

Networks are created through methods of :doc:`auto_workingband` object. 

Single Stub Tuner
----------------------

This is an example of how to design a single stub tuning network to match a 100ohm resistor to a 50 ohm environment. 

::

	con = mv.connect # for short-hand name for convenience
	
	
	Gamma0 = mv.zl_2_Gamma0(100)	
	load = wb.load(Gamma0)
	wb.shunt_open()

Double Stub Tuner
--------------------

Optimizing 
-------------------

