.. _installation:

Installation
****************


Requirements
-----------------

The requirements are basically a python environment setup to do numerical/scientific computing. If you are new to Python development, I recommend you install a pre-built scientific python IDE like pythonxy. This will install all requirements, as well as provide a nice environment to get started in. If you dont want use Pythonxy, there is a list of requirements at end of this section.

NOTE: if you want to use mwavepy for instrument control you will need to install pyvisa manually. The link is given in List of Requirements section. Also, you may be interested in David Urso's Pythics module, for easy gui creation.

Install mwavepy
-----------------

There are three choices for installing mwavepy:

*    windows installer
*   python source package 
*    SVN version 

They can all be found here http://code.google.com/p/mwavepy/downloads/list

If you dont know how to install a python module and dont care to learn how, you want the windows installer.

If you know how to install a python package but aren't familiar with SVN then you want the Python source package . Examples, documentation, and installation instructions are provided in the the python package.

If you know how to use SVN, I recommend the SVN version because it has more features.

Linux-Specific
-------------------

For debian-based linux users who dont want to install Pythonxy, here is a one-shot line to install all requirements,

sudo apt-get install python-pyvisa python-numpy python-scipy::

 python-matplotlib ipython python

List of Requirements
------------------------

Here is a list of the requirements, Necessary:

*    python (>=2.6) http://www.python.org/
*    matplotlib (aka pylab) http://matplotlib.sourceforge.net/
*    numpy http://numpy.scipy.org/
*    scipy http://www.scipy.org/ ( provides tons of good stuff, check it out) 

Optional:

*    pyvisa http://pyvisa.sourceforge.net/pyvisa/ - for instrument control
*    ipython http://ipython.scipy.org/moin/ - for interactive shell
*    Pythics http://code.google.com/p/pythics - instrument control and gui creation 
