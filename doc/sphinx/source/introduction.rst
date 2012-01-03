.. _introduction:


.. currentmodule:: mwavepy.network

*******************
Introduction
*******************

This is a brief introduction to mwavepy usage, aimed at those who are familiar with python. All of the touchstone files used in these tutorials are provided along with this documentation, and are located in the directory ``./pyplots/`` (relative to this file).

Creating Networks from Touchstone Files
---------------------------------------

First, import mwavepy and name it something short, like 'mv'::

	>>> import mwavepy as mv

If this produces an error, please see :doc:`installation`. 

The most fundamental object in mwavepy is a n-port :class:`Network`. Most commonly, a :class:`Network` is constructed from data stored in a touchstone files, like so.::
	
	>>> short = mv.Network('short.s1p')
	>>> delay_short = mv.Network('delay_short.s1p')

Network Properties
-------------------------------------
	
The basic quantities associated with a :class:`Network` are provided by the 
following properties :

* :attr:`Network.s` : Scattering Parameter matrix. 
* :attr:`Network.z0`  : Characterisic Impedance matrix.
* :attr:`Network.frequency`  : Frequency Object. 


These properties are stored as complex numpy.ndarray's. Note that if you are using 
Ipython, then other properties and methods of the :class:`Network` class, can be 
'tabbed' out. Amongst other things, the methods of the :class:`Network` class provide
convenient ways to plot components of the s-parameters, below is a 
non-exhaustive list of common plotting commands,

* :func:`Network.plot_s_db` : plot magnitude of s-parameters in log scale
* :func:`Network.plot_s_deg` : plot phase of s-parameters in degrees
* :func:`Network.plot_s_smith` : plot complex s-parameters on Smith Chart

For example, to create a 2-port :class:`Network` from a touchstone file,
and then plot all s-parameters on the Smith Chart.

.. plot:: ./pyplots/introduction/simple_plot.py
   :include-source:

For more detailed information about plotting see :doc:`plotting`.   


Element-wise Operations 
-------------------------------------
	
Element-wise mathematical operations on the scattering parameter matrices are accessible through overloaded operators::

	>>> short + delay_short
	>>> short - delay_short 
	>>> short / delay_short 
	>>> short * delay_short

All of these operations return :class:`Network` types, so all  methods and properties of a :class:`Network` are available on the result.  For example, the difference operation ('-') can be used to calculate the complex distance between two networks ::
	
	>>> difference = (short- delay_short)

Because this returns :class:`Network` type, the distance is accessed through the :attr:`Network.s` property. The plotting methods of the :class:`Network` type can also be used. So to plot the magnitude of the complex difference between the networks `short` and `delay_short`::

	>>> (short - delay_short).plot_s_mag()

Another use of operators is calculating the phase difference using the division operator. This can be done ::
	
	>>> (delay_short/short).plot_s_deg()
	
	
Cascading and Embeding Operations 
----------------------------------------------
Cascading and de-embeding 2-port Networks is done so frequently, that it can also be done though operators as well. The cascade function is called by the power operator,  ``**``, and the de-embedding operation is accomplished by cascading the inverse of a network, which is implemented by the property :attr:`Network.inv`. Given the following Networks::

	>>> line = mv.Network('line.s2p')
	>>> short = mv.Network('short.s1p')
	
To calculate a new network which is the cascaded connection of the two individual Networks ``line`` and ``short``::
	
	>>> delay_short = line ** short

or to de-embed the *short* from *delay_short*::

	>>> short = line.inv ** delay_short



Connecting Multi-ports 
------------------------
**mwavepy** supports the connection of arbitrary ports of N-port networks. It accomplishes this using an algorithm call sub-network growth[#]_. This algorithm, which is available through the function :func:`connect`, takes into account port impedances. Terminating one port of a ideal 3-way splitter can be done like so::

	>>> tee = mv.Network('tee.s3p')
	>>> delay_short = mv.Network('delay_short.s1p')

to connect port '1' of the tee, to port 0 of the delay short::

	>>> terminated_tee = mv.connect(tee,1,delay_short,0)
	
Sub-Networks
---------------------
Frequently, the one-port s-parameters of a multiport network's are of interest. These can be accessed by properties such as::

	>>> port1_return = line.s11
	>>> port1_insertion = line.s21

References
----------	
.. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

