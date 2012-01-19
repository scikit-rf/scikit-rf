.. _installation:

****************
Installation
****************
.. contents::

Introduction
-----------------

The requirements to run skrf are basically a python environment setup to do numerical/scientific computing. If you are new to  Python_ development, you may want to install a pre-built scientific python IDE like pythonxy_. This will install all requirements, as well as provide a nice environment to get started in. If you dont want use pythonxy_, you see `Requirements`_.

**Note:** If you want to use skrf for instrument control you will need to install pyvisa. You may also be interested in Pythics, which provides a simple way to build interfaces to virtual instruments. Links is provided in `Requirements`_ section. 

skrf Installation 
-----------------------------

Once the requirements are installed, there are two choices for installing skrf:

*    windows installer
*   python source package 

They can all be found at http://code.google.com/p/skrf/downloads/list

If you dont know how to install a python module and dont care to learn how, you want the windows installer. Otherwise, I recommend the python source package because examples, documentation, and installation instructions are provided with the the python package.

The current version can be accessed through `SVN <http://code.google.com/p/skrf/source/checkout>`_. This is mainly of interest for developers, and is not stable most of the time.




Requirements
------------

Debian-Based Linux
======================

For debian-based linux users who dont want to install pythonxy_, here is a one-shot line to install all requirements,::

	sudo apt-get install python-pyvisa python-numpy python-scipy python-matplotlib ipython python


Necessary
=============

*    python (>=2.6) http://www.python.org/
*    matplotlib (aka pylab) http://matplotlib.sourceforge.net/
*    numpy http://numpy.scipy.org/
*    scipy http://www.scipy.org/ ( provides tons of good stuff, check it out) 

Optional
==========

*    ipython http://ipython.scipy.org/moin/ - for interactive shell
*    pyvisa http://pyvisa.sourceforge.net/pyvisa/ - for instrument control
*    Pythics http://code.google.com/p/pythics - instrument control and gui creation 


.. _Python: http://www.python.org/
.. _pythonxy: http://code.google.com/p/pythonxy/
