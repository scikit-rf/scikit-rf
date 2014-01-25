.. _calibration:

Calibration
***************
.. currentmodule:: skrf.calibration.calibration


Introduction
---------------

This tutorial describes how to use **skrf** to calibrate data taken from a VNA. For an introduction to VNA calibration see this article by Rumiantsev and Ridler [1]_, for an outline of different calibration algorithms see Doug Ryttings presentation [2]_. If you like to read books, then you may want to checkout [3]_ .

What follows is are various examples of how to calibrate a device under test (DUT), assuming you have measured an acceptable set of standards, and have a coresponding set ideal responses. This may be reffered to as *offline* calibration, because it is not occuring onboard the VNA itself. One benefit of this technique is that it provides maximum flexibilty for non-conventional calibrations, and preserves all raw data.


Creating a Calibration
----------------------------

Calibrations are performed through a :class:`Calibration` class. In General, 
all :class:`Calibration` objects require at least two arguments:

*   a list of measured :class:`~skrf.network.Network`'s
*   a list of ideal :class:`~skrf.network.Network`'s

The :class:`~skrf.network.Network` elements in each list must all be similar (same number of ports, frequency info, etc) and must be aligned to each other, meaning the first element of ideals list must correspond to the first element of measured list.


The following algorithms are supported in part or in full. See each class
for details.

* :class:`OnePort`
* :class:`SOLT`
* :class:`EightTerm`
* :class:`TRL`


One-Port Example
-----------------

This example is written to be instructive, not concise. To 
construct a one-port calibration you need to have measured at least 
three standards and have their known *ideal* responses in the form of
:class:`~skrf.network.Network`s. These :class:`~skrf.network.Network`
can be created from touchstone files, a format all modern VNA's support.

In the following script  the measured and ideal touchstone files for 
a conventional short-open-load (SOL) calkit are in folders `measured/` 
and `ideal/`, respectively. These are used to create a OnePort Calibration and correct a measured DUT ::

	import skrf as rf
	from skrf.calibration import OnePort
	
	## created necessary data for Calibration class
	
	# a list of Network types, holding 'ideal' responses
	my_ideals = [\
	        rf.Network('ideal/short.s1p'),
	        rf.Network('ideal/open.s1p'),
	        rf.Network('ideal/load.s1p'),
	        ]
	
	# a list of Network types, holding 'measured' responses
	my_measured = [\
	        rf.Network('measured/short.s1p'),
	        rf.Network('measured/open.s1p'),
	        rf.Network('measured/load.s1p'),
	        ]
	
	## create a Calibration instance
	cal = rf.OnePort(\
	        ideals = my_ideals,
	        measured = my_measured,
	        )
	
	
	## run, and apply calibration to a DUT
	
	# run calibration algorithm
	cal.run() 
	
	# apply it to a dut
	dut = rf.Network('my_dut.s1p')
	dut_caled = cal.apply_cal(dut)
	dut_caled.name =  dut.name + ' corrected'
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

Concise One-port
-----------------

This example achieves the same task as the one above except in a 
more concise *programmatic* way. ::

    import skrf as rf
    from skrf.calibration import OnePort
    
    my_ideals = rf.load_all_touchstones_in_dir('ideals/')
    my_measured = rf.load_all_touchstones_in_dir('measured/')
    
    
    ## create a Calibration instance
    cal = rf.OnePort(\
	    ideals = [my_ideals[k] for k in ['short','open','load']],
	    measured = [my_measured[k] for k in ['short','open','load']],
	    )
    
    ## what you do with 'cal' may  may be similar to above example

Two-port
---------

Naturally, two-port calibration is more involved than one-port. **skrf**
supports a few different two-port algorithms. The traditional :class:`SOLT` algorithm uses the 12-term error model. This algorithms is straightforward, and similar to the OnePort example.

The :class:`EightTerm` calibration is based on the algorithm described in [4]_, by R.A. Speciale. It can  be constructed from any number of standards, providing that some fundamental constraints are met. In short, you need three two-port standards; one must be transmissive, and one must provide a known impedance and be reflective.
Note, that the word *8-term* is used in the literature to describe a specific error model used by a variety of calibration algorihtms, like TRL, LRM, etc. The :class`EightTerm` class, is an implementation of the algorithm cited above, which does not use any self-calibration. 

One important detail of using the 8-term error model formulation is that switch-terms may need to be measured in order to achieve a high quality calibration (thanks to Dylan Williams for pointing this out). These are described next.

Switch-terms
++++++++++++++++++++++++

Originally described by Roger Marks [5]_ , switch-terms account for the fact that the 8-term (aka *error-box* ) model is overly simplified.  The two error networks change slightly depending on which port is being excited. This is due to the internal switch within the VNA.

Switch terms can be directly measured with a custom measurement configuration on the VNA itself. **skrf** has support for measuring switch terms in the :mod:`~skrf.vi.vna` module, see the HP8510's :attr:`~skrf.vi.vna.HP8510C.switch_terms`, or PNA's  :func:`~skrf.vi.vna.PNA.get_switch_terms` . Without switch-term measurements, your calibration quality will vary depending on properties of you VNA.


SOLT Example
-------------------

Two-port calibration is accomplished in an identical way to one-port, except all the standards are two-port networks. This is even true of reflective standards (S21=S12=0). So if you measure reflective standards you must measure two of them simultaneously, and store information in a two-port. For example, connect a short to port-1 and a load to port-2, and save a two-port measurement as 'short,load.s2p' or similar::

    import skrf as rf
    from skrf.calibration import SOLT
    
    
    
    # a list of Network types, holding 'ideal' responses
    my_ideals = [
	    rf.Network('ideal/thru.s2p'),
	    rf.Network('ideal/short, short.s2p'),
	    rf.Network('ideal/open, open.s2p'),
	    rf.Network('ideal/load, load.s2p'),
	    ]
    
    # a list of Network types, holding 'measured' responses
    my_measured = [
	    rf.Network('measured/thru.s2p'),
	    rf.Network('measured/short, short.s2p'),
	    rf.Network('measured/open, open.s2p'),
	    rf.Network('measured/load, load.s2p'),
	    ]
    
    
    ## create a SOLT instance
    cal = SOLT(
	    ideals = my_ideals,
	    measured = my_measured,
	    )
    
    
    ## run, and apply calibration to a DUT
    
    # run calibration algorithm
    cal.run() 
    
    # apply it to a dut
    dut = rf.Network('my_dut.s2p')
    dut_caled = cal.apply_cal(dut)
    
    # plot results
    dut_caled.plot_s_db()
    # save results 
    dut_caled.write_touchstone()

Using one-port ideals in two-port Calibration
++++++++++++++++++++++++++++++++++++++++++++++

Commonly, you have data for ideal data for reflective standards in the form of one-port touchstone files (ie `.s1p`). To use this with skrf's two-port calibration method you need to create a two-port network that is a composite of the two networks. The  function :func:`~skrf.network.two_port_reflect` does this::
    
    short = rf.Network('ideals/short.s1p')
    shorts = rf.two_port_reflect(short, short)




Saving and Recalling a Calibration
-----------------------------------

Like other **skrf** objects, :class:`Calibration`'s  can be written-to  and 
read-from disk. Writing  can be accomplished  by using :func:`Calibration.write`, 
or :func:`rf.write() <skrf.io.general.write>`, and reading is done with :func:`rf.read() <skrf.io.general.read>`. These functions rely on pickling, therefore they are not recomended for long-term data storage. Currently there is no way to achieve long term storage of a Calibration object, other than saving the script used to generate it.



.. rubric:: Bibliography


.. [1] http://www.cnam.umd.edu/anlage/Microwave%20Measurements%20for%20Personal%20Web%20Site/Rumiantsev%20VNA%20Calibration%20IEEE%20Mag%20June%202008%20page%2086.pdf


.. [2] http://www-ee.uta.edu/online/adavis/ee5349/NA_Error_Models_and_Cal_Methods.pdf

.. [3] J. P. Dunsmore, Handbook of microwave component measurements: with advanced vna techniques. Hoboken, N.J: Wiley, 2012.


.. [4] Speciale, R.A.; , "A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors," Microwave Theory and Techniques, IEEE Transactions on , vol.25, no.12, pp. 1100- 1115, Dec 1977. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1129282&isnumber=25047 


.. [5] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931  


