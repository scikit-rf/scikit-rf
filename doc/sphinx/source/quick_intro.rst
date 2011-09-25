.. _quick-intro:

Quick Introduction
*********************

This quick intro of basic mwavepy usage. It is aimed at those who are familiar with python, or are impatient. If you want a slower introduction, see the :doc:`slow_intro`.

Loading Touchstone Files
++++++++++++++++++++++++++

First, import mwavepy and name it something short, like ''mv''::

	import mwavepy as mv

The most fundamental object mwavpey is a n-port *Network*. Commonly a Network is constructed from data stored in a touchstone files, like so.::
	
	short = mv.Network ('short.s1p')
	delay_short = mv.Network ('delay_short.s1p')

Important Properties
+++++++++++++++++++++++++
	
The important qualities of a *Network* are provided by the properties:

* **s**: Scattering Parameter matrix. 
* **frequency**: Frequency Object. 
* **z0**: Characterisic Impedance matrix.

Element-wise Operations (Linear)
++++++++++++++++++++++++++++++++
	
Simple element-wise mathematical operations on the scattering parameter matrices are accesable through overloaded operators::

	short + delay_short
	short - delay_short 
	short / delay_short 
	short * delay_short

These have various uses. For example, the difference operation returns a network that represents the complex distance between two networks. This can be used to calculate the euclidean norm between two networks like ::
	
	(short- delay_short).s_mag

or you can plot it::

	(short - delay_short).plot_s_mag()

Another use is calculating or plotting de-trended phase using the division operator. This can be done by::
	
	detrended_phase = (delay_short/short).s_deg
	(delay_short/short).plot_s_deg()
	
	
Cascading and Embeding Operations (Non-linear)
++++++++++++++++++++++++++++++++++++++++++++++++
Cascading and de-embeding 2-port Networks is done so frequently, that it can also be done though operators. The cascade function is called by the power operator,  ``**``, and the de-embed function is done by cascading the inverse of a network, which is implemented by the property ``inv``. Given the following Networks::

	cable = mv.Network('cable.s2p')
	dut = mv.Network('dut.s1p')
	
Perhaps we want to calculate a new network which is the cascaded connection of the two individual Networks *cable* and *dut*::
	
	cable_and_dut = cable ** dut

or maybe we want to de-embed the *cable* from *cable_and_dut*::

	dut = cable.inv ** cable_and_dut

You can check my functions for consistency using the equality operator ::

	dut == cable.inv (cable ** dut)

if you want to de-embed from the other side you can use the flip() function provided by the Network class::

	dut ** (cable.inv).flip()

Sub Networks
++++++++++++++++++++++++++
Frequently, the individual responses of a higher order network are of interest. Network type provide way quick access like so::

	reflection_off_cable = cable.s11
	transmission_through_cable = cable.s21


Connecting Multi-ports 
+++++++++++++++++++++++++
**mwavepy** supports the connection of arbitrary ports of N-port networks. It does this using an algorithm call sub-network growth. This connection process takes into account port impedances.
Terminating one port of a ideal 3-way splitter can be done like so::

	tee = mv.Network('tee.s3p')
	delay_short = mv.Network('delay_short.s1p')

to connect port '1' of the tee, to port 0 of the delay short::

	terminated_tee = mv.connect(tee,1,delay_short,0)
