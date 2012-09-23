.. _introduction:

*******************
Introduction
*******************
.. currentmodule:: skrf.network
.. contents::

This is a brief introduction to skrf, aimed at those who are familiar with python. If you are unfamiliar with python, please see scipy's `Getting Started <http://www.scipy.org/Getting_Started>`_ . This tutorial is most easily followed by using the ipython_ shell with the `--pylab` flag. 

**Note:** All of the scripts and touchstone files used in these tutorials are provided along with this source code in the directory ``doc/source/pyplots/`` (relative to the scikit-rf root).

Creating Networks 
------------------

For this tutorial, and the rest of the scikit-rf documentation, it is  assumed that skrf has been imported as ``rf``. Whether or not you follow this convention in your own code is up to you::

	>>> import skrf as rf

If this produces an error, please see :doc:`installation`. 

skrf provides an object for a n-port microwave :class:`Network`. Most commonly, a :class:`Network` is constructed from data stored in a touchstone files, like so ::
	
	>>> short = rf.Network('short.s1p')
	>>> delay_short = rf.Network('delay_short.s1p')

Once created, the :class:`Network` object will print out a short description if entered onto the command line::
	
	>>> short
	1-Port Network.  75-110 GHz.  201 points. z0=[ 50.]

Basic Network Properties
-------------------------
	
The basic attributes of a microwave :class:`Network` are provided by the 
following properties :

* :attr:`Network.s` : Scattering Parameter matrix. 
* :attr:`Network.z0`  : Characterisic Impedance matrix.
* :attr:`Network.frequency`  : Frequency Object. 


These properties are represented internally as complex numpy.ndarray's, (and the indexing starts at 0!). The :class:`Network` class has numerous other properties and methods which can found in the  :class:`Network` docs. If you are using Ipython, then these properties and methods can be 'tabbed' out on the command line. Amongst other things, the methods of the :class:`Network` class provide convenient ways to plot components of the s-parameters, below is a short list of common plotting commands,

* :func:`Network.plot_s_db` : plot magnitude of s-parameters in log scale
* :func:`Network.plot_s_deg` : plot phase of s-parameters in degrees
* :func:`Network.plot_s_smith` : plot complex s-parameters on Smith Chart

For example, to create a 2-port :class:`Network` from a touchstone file,
and then plot all s-parameters on the Smith Chart.


.. plot:: ../pyplots/introduction/simple_plot.py
   :include-source:

For more detailed information about plotting see :doc:`plotting`.   

Network Operators
-------------------

Element-wise Operations 
=========================
	
Element-wise mathematical operations on the scattering parameter matrices are accessible through overloaded operators::

	>>> short + delay_short
	>>> short - delay_short 
	>>> short / delay_short 
	>>> short * delay_short

All of these operations return :class:`Network` types, so the methods and properties of a :class:`Network` are available on the result.  For example, the difference operation ('-') can be used to calculate the complex distance between two networks ::
	
	>>> difference = (short- delay_short)

The result is stored in the returned :class:`Network` object's  :attr:`Network.s` parameter matrix.  So, to plot the magnitude of the complex difference between the networks `short` and `delay_short`::

	>>> (short - delay_short).plot_s_mag()

Another common application is calculating the phase difference using the division operator. This can be done ::
	
	>>> (delay_short/short).plot_s_deg()
	
	
Cascading and Embeding Operations 
==================================================
Cascading and de-embeding 2-port Networks can also be done though operators. The :func:`cascade` function can be accessed through the power operator,  ``**``. To calculate a new network which is the cascaded connection of the two individual Networks ``line`` and ``short``, ::

	>>> line = rf.Network('line.s2p')
	>>> short = rf.Network('short.s1p')	
	>>> delay_short = line ** short

De-embedding  can be accomplished either by using the  :func:`de_embed` function, or  by cascading the *inverse* of a network. The inverse of a network is accesible through the property :attr:`Network.inv`. So,  to de-embed the *short* from *delay_short*::

	>>> short = line.inv ** delay_short



Connecting Multi-ports 
------------------------
**skrf** supports the connection of arbitrary ports of N-port networks. It accomplishes this using an algorithm call sub-network growth [#]_. This algorithm, which is available through the function :func:`connect`, takes into account port impedances. Terminating one port of a ideal 3-way splitter can be done like so::

	>>> tee = rf.Network('tee.s3p')
	>>> delay_short = rf.Network('delay_short.s1p')

to connect port '1' of the tee, to port 0 of the delay short::

	>>> terminated_tee = rf.connect(tee,1,delay_short,0)
	
Sub-Networks
---------------------
Frequently, the one-port s-parameters of a multiport network's are of interest. These can be accessed by theh sub-network properties, which return one-port :class:`Network` objects ::

	>>> port1_return = line.s11
	>>> port1_insertion = line.s21
	
	
Convenience Functions
---------------------
Frequently there is an entire directory of touchstone files that need to be analyzed. The function :func:`load_all_touchstones` is meant deal with this scenario. It takes a string representing the directory,  and returns a dictionary type with keys equal to the touchstone filenames, and values equal to :class:`Network` types::
	
	>>> ntwk_dict = rf.load_all_touchstones('.')
	{'delay_short': 1-Port Network.  75-110 GHz.  201 points. z0=[ 50.],
	'line': 2-Port Network.  75-110 GHz.  201 points. z0=[ 50.  50.],
	'ring slot': 2-Port Network.  75-110 GHz.  201 points. z0=[ 50.  50.],
	'short': 1-Port Network.  75-110 GHz.  201 points. z0=[ 50.]}
	
Other convenient functions, and pre-initialized objects are located in the 
:mod:`convenience` module


References
----------	
.. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

.. _ipython: http://ipython.scipy.org/moin/	
