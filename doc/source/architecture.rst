.. _architecture:


Architecture
**********************



Module Layout and Inheritance
------------------------------
.. figure::  images/classInheretanceOutline.eps
   :align:   center
   :width:	800

Individual Class Architectures
-------------------------------
.. figure::  images/classOutline.eps
   :align:   center
   :width:	800

Frequency
++++++++++++

The frequency object was created to make storing and manipulating frequency information easier and more rigid. A major convenience this class provides is the acounting of the frequency vector's unit. Other objects, such as Network, and Calibration require a frequency vector to be meaningful. This vector is commonly referenced when a plot is generated, which one generally doesnt was in units of Hz. If the Frequency object did not exist other objects which require frequency information would have to implement the unit and multiplier bagage. 

.. figure::  images/frequency.eps
   :align:   center
   :width:	800



Network
++++++++++
.. figure::  images/network.eps
   :align:   center
   :width:	800


touchstone
++++++++++++

The standard file format used to store data retrieved from Vector Network Analyzers (VNAs) is the touchstone file format. This file contains all relevent data of a measured network such as frequency info, network parameters (s, y,z, etc), and port impedance.

WorkingBand
+++++++++++++

.. figure::  images/workingBand.eps
   :align:   center
   :width:	800


Calibration
+++++++++++++
.. figure::  images/calibration.eps
   :align:   center
   :width:	800

