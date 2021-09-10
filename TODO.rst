TODO list 
============

This is a list of TODO's for skrf. Feel free to add to this list via 
pull request. 

Network 
-----------
* comments support
* isolate features in tests (ie dont use touchstones to test cascading)
* make operators symetric 
* add interpolation of arbitrary network parameters
* make interpolation functions more clear
* incorperate support for c, and connect_s_fast, or remove
* ABCD parameters

NetworkSet
------------
* add tests
* fix pickling problem. 

IO
-----
* Citi files
* .nmf or similar data-base like set
* touchstone - ansoft style z0/gamma
* pickling of media objects


Calibration
------------
* TRL
* simplify module, consolidate files, simplify docs into one page
* fixed biased error to not depend on names of calibrations
* clean up CalibrationSet, maybe merge into Calibraiton class
* use NetworkSets to implement biased vs. unbiased errors

Frequency 
-----------
* add write() method

vi 
--------------------
* re-write VNA's coherently 
* use consistent method names so VNA's are inter-changable as much as possible
* make a generic VNA that uses IDN? to load the correct class (durrants idea)

Other
------
* switch to BSD license similar to scipy/numpy/etc
* use doctests or ipython doctests
* change ininstance() calls so that the we dont depend on module name

 -- see Network.frequency.setter upon change of skrf to skrf2 or similar

plotting
-------------
* have plots return the plot object
