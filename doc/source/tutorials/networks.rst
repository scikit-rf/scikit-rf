.. _networks:

*******************
Networks
*******************
.. currentmodule:: skrf.network
.. contents::



.. ipython::
	:suppress:
	
	
	In [144]: from pylab import *
	
	In [145]: ion()
	
	In [146]: rcParams['savefig.dpi'] =120
	
	In [147]: rcParams['figure.figsize'] = [4,3]
	
	In [147]: rcParams['figure.subplot.left'] = 0.15
	
	In [147]: clf()
	


Introduction
-------------------------
For this tutorial, and the rest of the scikit-rf documentation, it is  assumed that **skrf** has been imported as ``rf``. Whether or not you follow this convention in your own code is up to you.


.. ipython::

  In [138]: import skrf as rf

If this produces an error, please see :doc:`installation`.  The code in this tutorial assumes that you are in the directory ``scikit-rf/doc``.

Creating Networks
-------------------------
**skrf** provides an object for a N-port microwave :class:`Network`. A :class:`Network` can be created in a number of ways. One way is from data stored in a touchstone file.

.. ipython::
			
	In [139]: ring_slot = rf.Network('../skrf/data/ring slot.s2p')

 
	
A short description of the network will be printed out if entered onto the command line
	
.. ipython::
	
	In [1]: ring_slot

Networks can also be created from a pickled Network (written by :func:`Network.write`), 

.. ipython::
	
	In [139]: ring_slot = rf.Network('../skrf/data/ring slot.ntwk') 
	
or from directly passing values for the frequency, s-paramters  and z0. 

.. ipython::
	
	In [1]: custom_ntwk = rf.Network(f = [1,2,3], s= [-1, 1j, 0], z0=50)
	# `f` is interpreted in units of 'ghz'

Seen :func:`Network.__init__`  for more informaition on network creation.

Network Basics
-------------------------
	
The basic attributes of a microwave :class:`Network` are provided by the 
following properties :

* :attr:`Network.s` : Scattering Parameter matrix. 
* :attr:`Network.z0`  : Port Characteristic Impedance matrix.
* :attr:`Network.frequency`  : Frequency Object. 

All of the network parameters are represented internally as complex :class:`numpy.ndarray` 's of shape *FxNxN*, where *F* is the number of frequency points and *N* is the number of ports.

.. ipython::
			
	In [139]: shape(ring_slot.s)

Note that the indexing starts at 0, so the first 10 values of :math:`S_{11}` can be accessed with

.. ipython::
			
	In [139]: ring_slot.s[:10,0,0]


The :class:`Network` object has numerous other properties and methods which can found in the :class:`Network` docstring. If you are using IPython, then these properties and methods can be 'tabbed' out on the command line. 


.. ipython::
	
	@verbatim
	In [1]: short.s<TAB>
	rf.data.line.s              rf.data.line.s_arcl         rf.data.line.s_im
	rf.data.line.s11            rf.data.line.s_arcl_unwrap  rf.data.line.s_mag
	...

.. note:: 

	Although this tutorial focuses on s-parametes, other  network representations such as Impedance (:attr:`Network.z`) and Admittance Parameters (:attr:`Network.y`) are available as well, see `Impedance and Admittance Parameters`_ .
	
Amongst other things, the methods of the :class:`Network` class provide convenient ways to plot components of the network parameters, 

* :func:`Network.plot_s_db` : plot magnitude of s-parameters in log scale
* :func:`Network.plot_s_deg` : plot phase of s-parameters in degrees
* :func:`Network.plot_s_smith` : plot complex s-parameters on Smith Chart
* ...

To plot all four s-parameters of the ``ring_slot`` on the Smith Chart.

.. ipython::

   @savefig ring_slot,smith.png 
   In [151]: ring_slot.plot_s_smith();


Or plot a pair of s-parameters individually, in log magnitude 

.. ipython::

   In [153]: figure();
   
   In [153]: ring_slot.plot_s_db(m=1, n=0);	# s21
   
   @savefig ring_slot,db.png 
   In [153]: ring_slot.plot_s_db(m=0, n=0); # s11

For more detailed information about plotting see :doc:`plotting`.   

Network Operators
-------------------

Linear Operations 
=========================
	
Element-wise mathematical operations on the scattering parameter matrices are accessible through overloaded operators. To illustrate their usage, load a couple  Networks stored in the ``data`` module. 

.. ipython::
	
	In [21]: short = rf.data.wr2p2_short

	In [22]: delayshort = rf.data.wr2p2_delayshort

	In [22]: short - delayshort

	In [22]: short + delayshort

	In [22]: short * delayshort

	In [22]: short / delayshort
	
	In [22]: short / delayshort



All of these operations return :class:`Network` types, so the methods and properties of a :class:`Network` are available on the result.  For example, to plot the complex difference  between  ``short`` and ``delay_short``,
	
.. ipython::
	
	In [21]: figure();
	
	In [21]: difference = (short- delayshort)
	
	@savefig operator_illustration,difference.png
	In [21]: difference.plot_s_mag()


Another common application is calculating the phase difference using the division operator,
	
.. ipython::
	
	In [21]: figure();
	
	@savefig operator_illustration,division.png
	In [21]: (delayshort/short).plot_s_deg()
	
Linear operators can also be used with scalars or an :class:`numpy.ndarray`  that ais the same length as the :class:`Network`. 

.. ipython::
	
	In [21]: open = (short*-1)

	In [21]: open.s[:3,...]
	
	In [21]: rando =  open *rand(len(open))
	
	In [21]: rando.s[:3,...]
	
Note that if you multiply a Network by an :class:`numpy.ndarray`  be sure to place the array on right side.

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


Connecting Multi-ports 
------------------------
**skrf** supports the connection of arbitrary ports of N-port networks. It accomplishes this using an algorithm called sub-network growth [#]_,  available through the function :func:`connect`. Terminating one port of an ideal 3-way splitter can be done like so,

.. ipython::
	
	In [21]: tee = rf.Network('../skrf/data/tee.s3p')
	

To connect port `1` of the tee, to port `0` of the delay short,

.. ipython::
	
	In [21]: terminated_tee = rf.connect(tee,1,delayshort,0)

Note that this function takes into account port impedances, and if connecting ports have different port impedances an appropriate impedance mismatch is inserted.
	
Interpolation and Stitching 
-----------------------------
A common need is to change the number of frequency points of a :class:`Network`. For instance, to use the operators and cascading functions the networks involved must have matching frequencies. If two networks have different frequency information, then an error will be raised, 

.. ipython::
	
	In [21]: line = rf.data.wr2p2_line.copy()
	
	In [21]: line1 = rf.data.wr2p2_line1.copy()
	
	In [21]: line1
	
	In [21]: line
	
	In [21]: line1+line
	
This problem can be solved by interpolating one of Networks, using :func:`Network.resample`. 

.. ipython::
	
	In [21]: line1
	
	In [21]: line1.resample(201)
	
	In [21]: line1
	
	In [21]: line1+line


A related application is the need to combine Networks which cover different frequency ranges. Two  Netwoks can be stitched together using :func:`stitch`, which  concatenates their s-parameter matrices along their frequency axis. To combine a WR-2.2 Network with a WR-1.5 Network, 
 
.. ipython::
	
	In [21]: from skrf.data import wr2p2_line, wr1p5_line
	
	In [21]: line = rf.stitch(wr2p2_line, wr1p5_line)
	
	In [21]: line
	



	
Reading and Writing 
------------------------------
While **skrf** supports reading and writing the touchstone file format, it also provides native IO capabilities for any skrf object through the functions :func:`~skrf.io.general.read` and :func:`~skrf.io.general.write`. These functions can also be called through the Network methods :func:`Network.read` and :func:`Network.write`. The Network constructor (:func:`Network.__init__` ) calls :func:`~skrf.io.general.read` implicitly if a skrf file is passed.

.. ipython::
	
	In [21]: line = rf.Network('../skrf/data/line.s2p')
	
	@verbatim
	In [21]: line.write() # write out Network using native IO
	line.ntwk
	
	@verbatim
	In [21]: rf.Netwrok('line.ntwk') # read Network using native IO

Frequently there is an entire directory of files that need to be analyzed. The function :func:`~skrf.io.general.read_all` is used to create objects from all files in a directory quickly. Given a directory of skrf-readable files, :func:`~skrf.io.general.read_all`  returns a :class:`dict`  with keys equal to the filenames, and values equal to objects. To load all **skrf** files in the ``skrf/data/`` directory which contain the string ``\'wr2p2\'``.
	
.. ipython::
	
	In [21]: dict_o_ntwks = rf.read_all('../skrf/data/', contains = 'wr2p2')
	
	In [21]: dict_o_ntwks
	
:func:`~skrf.io.general.read_all` has a companion function, :func:`~skrf.io.general.write_all` which takes a dictionary of **skrf** objects, and writes each object to an individual file. 
	
.. ipython::
	:verbatim:

	In [21]: rf.write_all(dict_o_ntwks, dir = '.')
	
	In [21]: ls
	wr2p2,delayshort.ntwk	wr2p2,line.ntwk		wr2p2,short.ntwk

It is also possible to write a dictionary of objects to a single file, by using :func:`~skrf.io.general.write`,

.. ipython::
	:verbatim:

	In [21]: rf.write('dict_o_ntwk.p', dict_o_ntwks)

	In [21]: ls
	dict_o_ntwk.p
	
A similar function :func:`~skrf.io.general.save_sesh`, can be used to 
save all **skrf** objects in the current namespace.


Impedance and Admittance Parameters	
------------------------------------
This tutorial focuses on s-parameters, but other network represenations are available as well. Impedance  and Admittance Parameters can be accessed through the parameters :attr:`Network.z` and :attr:`Network.y`, respectively. Scalar components of complex parameters, such as  :attr:`Network.z_re`, :attr:`Network.z_im` and plotting methods like :func:`Network.plot_z_mag` are available as well.

.. ipython::
	
	In [21]: ring_slot.z[:3,...]
	
	In [21]: figure();
	
	@savefig ring_slot_z_re.png
	In [21]: ring_slot.plot_z_im(m=1,n=0)
	


Creating Networks 'From Scratch'	
----------------------------------
A :class:`Network` can be created `from scratch` by  passing values of relevant properties as keyword arguments to the constructor,

.. ipython::
			
	In [139]: frequency = rf.Frequency(75,110,101,'ghz')
	
	In [139]: s = -1*ones(101)
	
	In [139]: wr10_short = rf.Network(frequency = frequency, s = s, z0 = 50 )

For more information creating Networks representing transmission line and lumped components, see the :mod:`~skrf.media` module.

.. ipython::
	:suppress:
	
	In [144]: %logstop skrf_introduction.py


Sub-Networks
---------------------
Frequently, the one-port s-parameters of a multiport network's are of interest. These can be accessed by the sub-network properties, which return one-port :class:`Network` objects,

.. ipython::
	
	In [21]: port1_return = line.s11
	
	In [21]: port1_return 

References
----------	
.. [#] Compton, R.C.; , "Perspectives in microwave circuit analysis," Circuits and Systems, 1989., Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16 Aug 1989. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

.. _ipython: http://ipython.scipy.org/moin/	
