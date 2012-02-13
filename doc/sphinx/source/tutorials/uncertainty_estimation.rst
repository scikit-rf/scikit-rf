.. _uncertainty_estimation:

***********************
Uncertainty Estimation
***********************
.. currentmodule:: skrf.networkSet
.. contents::


**scikit-rf** can be used to calculate uncertainty estimates given a set 
of networks. The :class:`~NetworkSet` object holds sets of networks and
provides automated methods for calculating and displaying uncertainty 
bounds.

Although the uncertainty esimation functions operate on any set of
networks, the topic of uncertianty estimation is frequently associated 
with calibration uncertianty. That is, how certain can one be in 
results of a calibrated measurement.



Simple Case
--------------

Assume that numerous touchstone files, representing redudant 
measurements of a single network are made, such as::

    In [24]: ls *ro*
        ro,0.s1p  ro,1.s1p  ro,2.s1p  ro,3.s1p  ro,4.s1p  ro,5.s1p


In case you are curious, the network in this example is an open 
rectangular waveguide radiating into freespace. The numerous touchstone 
files `ro,0.s1p` , `ro,1.s1p`, ... , are redundant measurements on which we
would like to calculate the mean response, with uncertainty bounds.

This is most easily done by constructing :class:`~NetworkSet` object. 
The fastest way to achieve this is to use the convenience function 
:func:`~skrf.network.load_all_touchstones`, which returns a dictionary 
with :class:`~skrf.network.Network` objects for values.

.. plot:: ../pyplots/uncertainty_estimation/uncertainty_in_network_set.py
   :include-source:


	
	
