
skrf.media tests
=================

Running Tests
-----------------
These tests are most easily run using pytest, like so ::
    
    pytest skrf/media/tests/

Using QUCS for test-cases
--------------------------

skrf's media classes are tested by comparison with those produced by qucs (http://qucs.sourceforge.net/). The directory `qucs_prj/` is a qucs project directory which contains the projects used to generate the various touchstone files that are considered correct. The easiest way to use this project in qucs is to symbolically link to the `qucs_prj` directory in your qucs directory (~/.qucs/ by default). So for example, if you are in this directory ::

    ln -s skrf/media/tests/qucs_prj ~/.qucs/

Once you simulate the test network in qucs, you can convert the output 
data into a touchstone file by using `qucsconv`, for example :: 

    qucsconv -if qucsdata -i resistor.dat -of touchstone  -o resistor,1ohm.s2p

This touchstone can then be loaded into skrf as a Network, and tests
can be run. 


Structure of tests
-------------------

The test module `test_all_construction.py` tests that all media classes can call network creation methods that are inherited from the `Media` class. 

In addition to this, each media class should have a seperate test module that tests its functionality. 



