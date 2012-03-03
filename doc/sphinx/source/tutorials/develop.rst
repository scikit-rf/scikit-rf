.. _develop:

***********************
Developing skrf
***********************

.. contents::


Introduction
--------------
Welcome to the skrf's developer docs! This page is for those who are interested in participating those who are interested in developing scikit-rf. 

Starting in February, 2012, skrf's codebase has been  versioned using `git <http://git-scm.com/>`_ , and hosted on github at   https://github.com/scikit-rf/scikit-rf. The easiest way to contribute to any part of scikit-rf is to create an account on github, but if you are not familiar with git, dont hesitate to contact me directly by email at arsenovic@virginia.edu. 

Contributing Code
-----------------
skrf uses the `Fork + Pull` collaborative development model.
Please see github's page on this for more information http://help.github.com/send-pull-requests/


Contribute Documentation
-------------------------
skrf's documentation is generated using `sphinx <http://sphinx.pocoo.org/>`_. The documentation source code is written using reStructed Text, and can be found in ``docs/sphinx/source/``. The reference documentation for the submodules, classes, and functions are documented following the `conventions <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ put forth by Numpy/Scipy. Improvements or new documentation is welcomed, and can be submitted using github as well.



Creating Tests
--------------
skrf employs the python module `unittest` for testing. The test's are located in ``skrf/testCases/``.
