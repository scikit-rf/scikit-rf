.. _example-matching_single_stub:
*********************************************
Circuit Design: Single Stub Matching Network
*********************************************

Introduction
-------------
This example illustrates a way to visualize the design space for a single stub matching network. The matching Network consists of a shunt and series stub arranged as shown below, (image taken from R.M. Weikle's Notes)        

.. figure::  ../images/single_stub_matching_diagram.png
   :align:   center 
   :width:	400
   
A single stub matching network can be designed to produce maximum power transfer to the load, at a single frequency. The matching network has two design parameters:
 
 * length of series tline
 * length of shunt tline
  
This script illustrates how to create a plot of return loss magnitude off the matched load, vs series and shunt line lengths. The optimal designs are then seen as the minima of a 2D surface.    

Script
------------
.. plot:: ./pyplots/examples/single_stub_design_optimization.py
   :include-source:





