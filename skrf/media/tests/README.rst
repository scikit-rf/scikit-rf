
skrf.media tests
=================

skrf's media classes are tested by comparison with those produced by qucs (http://qucs.sourceforge.net/). The directory `qucs_prj/` is a qucs project directory which contains the projects used to generate the various touchstone files that aconsidered  correct. The easiest way to use this open this project in qucs is to symbolically link to the `qucs_prj` directory in your qucs directory (~/.qucs/ by default). So for example ::

    ln -s qucs_prj ~/.qucs/


The test module `test_all_construction.py` tests that all media classes can call network creation methods that are inheritted from the `Media` class. 

In addition to this, each media class should have a seperate test module that tests its functionality. 



