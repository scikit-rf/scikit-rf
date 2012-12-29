TODO list 
============

This is a list of TODO's for skrf. Feel free to add to this list via 
pull request. 

Network 
-----------
* is_reciprocal, is_passive, is_symetric
* comments support
* isolate features in tests (ie dont use touchstones to test cascading)
* make operators symetric 
* add interpolation of arbitrary network parameters
* make interpolation functions more clear
* incorperate support for c, and connect_s_fast
* ABCD parameters

IO
-----
* Citi files
* .nmf or similar data-base like set
* touchstone - ansoft style z0/gamma
* touchstone - comments 
* pickling of media objects


Calibration
------------
* TRL
* simplify module, consolidate files
* fixed biased error to not depend on names of calibrations
* clean up CalibrationSet, maybe merge into Calibraiton class
* use NetworkSets to implement biased vs. unbiased errors

Frequency 
-----------
* add write() method 

Other
------
* switch to BSD license similar to scipy/numpy/etc
