.. _calibration:

Calibration
***************
.. contents::
Intro
---------------

This page describes how to use **skrf** to calibrate data taken from a VNA. The explanation of calibration theory and calibration kit design is beyond the scope of this  page. This page describes how to calibrate a device under test (DUT), assuming you have measured an acceptable set of standards, and have a coresponding set ideal responses.

skrf's calibration algorithm is generic in that it will work with any set of standards. If you supply more calibration standards than is needed, skrf will implement a simple least-squares solution.

Calibrations are performed through a Calibration class, which makes creating and working with calibrations easy. Since skrf-1.2 the Calibration class only requires two pieces of information:

*   a list of measured Networks
*   a list of ideal Networks 

The Network elements in each list must all be similar, (same #ports, same frequency info, etc) and must be aligned to each other, meaning the first element of ideals list must correspond to the first element of measured list.

Optionally, other information can be provided for explicitness, such as,

*    calibration type
*    frequency information
*    reciprocity of embedding networks
*    etc 

When this information is not provided skrf will determine it through inspection.

One-Port
--------------

See :doc:`example_oneport_calibration` for examples.

Below are (hopefully) self-explanatory examples of increasing complexity, which should illustrate, by example, how to make a calibration.
Simple One-port

This example is written to be instructive, not concise.::

	import skrf as rf
	
	
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
	cal = rf.Calibration(\
	        ideals = my_ideals,
	        measured = my_measured,
	        )
	
	
	## run, and apply calibration to a DUT
	
	# run calibration algorithm
	cal.run() 
	
	# apply it to a dut
	dut = rf.Network('my_dut.s1p')
	dut_caled = cal.apply_cal(dut)
	
	# plot results
	dut_caled.plot_s_db()
	# save results 
	dut_caled.write_touchstone()

Concise One-port

This example is meant to be the same as the first except more concise.::

    import skrf as rf
    
    my_ideals = rf.load_all_touchstones_in_dir('ideals/')
    my_measured = rf.load_all_touchstones_in_dir('measured/')
    
    
    ## create a Calibration instance
    cal = rf.Calibration(\
	    ideals = [my_ideals[k] for k in ['short','open','load']],
	    measured = [my_measured[k] for k in ['short','open','load']],
	    )
    
    ## what you do with 'cal' may  may be similar to above example

Two-port
---------

Two-port calibration is more involved than one-port. skrf supports two-port calibration using a 8-term error model based on the algorithm described in [#]_, by R.A. Speciale.

Like the one-port algorithm, the two-port calibration can handle any number of standards, providing that some fundamental constraints are met. In short, you need three two-port standards; one must be transmissive, and one must provide a known impedance and be reflective.

One draw-back of using the 8-term error model formulation (which is the same formulation used in TRL) is that switch-terms may need to be measured in order to achieve a high quality calibration (this was pointed out to me by Dylan Williams).

A note on switch-terms
++++++++++++++++++++++++

Switch-terms are explained in a paper by Roger Marks  [#]_. Basically, switch-terms account for the fact that the error networks change slightly depending on which port is being excited. This is due to the hardware of the VNA.

So how do you measure switch terms? With a custom measurement configuration on the VNA itself. mwavpey has support for switch terms for the HP8510C class, which you can use or extend to different VNA. Without switch-term measurements, your calibration quality will vary depending on properties of you VNA.

See :doc:`example_twoport_calibration` for and example

Simple Two Port
-------------------

Two-port calibration is accomplished in an identical way to one-port, except all the standards are two-port networks. This is even true of reflective standards (S21=S12=0). So if you measure reflective standards you must measure two of them simultaneously, and store information in a two-port. For example, connect a short to port-1 and a load to port-2, and save a two-port measurement as 'short,load.s2p' or similar::

    import skrf as rf
    
    
    ## created necessary data for Calibration class
    
    # a list of Network types, holding 'ideal' responses
    my_ideals = [\
	    rf.Network('ideal/thru.s2p'),
	    rf.Network('ideal/line.s2p'),
	    rf.Network('ideal/short, short.s2p'),
	    ]
    
    # a list of Network types, holding 'measured' responses
    my_measured = [\
	    rf.Network('measured/thru.s2p'),
	    rf.Network('measured/line.s2p'),
	    rf.Network('measured/short, short.s2p'),
	    ]
    
    
    ## create a Calibration instance
    cal = rf.Calibration(\
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

Using s1p ideals in two-port calibration
++++++++++++++++++++++++++++++++++++++++++

Commonly, you have data for ideal data for reflective standards in the form of one-port touchstone files (ie s1p). To use this with skrf's two-port calibration method you need to create a two-port network that is a composite of the two networks. There is a function in the WorkingBand Class which will do this for you, called two_port_reflect.::
    
    short = rf.Network('ideals/short.s1p')
    load = rf.Network('ideals/load.s1p')
    short_load = rf.two_port_reflect(short, load)

.. rubric:: Bibliography

.. [#] Speciale, R.A.; , "A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors," Microwave Theory and Techniques, IEEE Transactions on , vol.25, no.12, pp. 1100- 1115, Dec 1977. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1129282&isnumber=25047 


.. [#] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931  
