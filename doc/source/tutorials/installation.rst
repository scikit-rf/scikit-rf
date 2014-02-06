.. _installation:

****************
Installation
****************


Introduction
-----------------

The requirements to run **skrf** are basically a Python_ environment setup to do numerical/scientific computing. If you are new to   development, you may want to install a pre-built scientific python IDE like Enthought's `Canopy <https://www.enthought.com/products/canopy/>`_. This will install most  requirements, as well as provide a nice environment to get started in. If you dont want use Canopy then see `Requirements`_.


.. note:: 

	If you want to use **skrf** for instrument control you will need to install `pyvisa <http://pyvisa.sourceforge.net/pyvisa/>`_ as well as the NI-GPIB drivers. 

**skrf** Installation 
-----------------------------

Once the requirements are installed, there are two choices for installing **skrf**:

*    windows installer
*   python source package 

These can be found at http://scikit-rf.org/download.html

If you dont know how to install a python module and dont care to learn how, you want the windows installer. 

The current version can be accessed through `github  <https://github.com/scikit-rf/scikit-rf>`_. This is mainly of interest for developers.

Testing Installation 
----------------------
If you can import **skrf** and dont recieve an error, then installation was succesful.

.. ipython::

	In [138]: import skrf as rf
  
Bingo. If instead you get an error like this, 

.. ipython::
	:verbatim:
	
	In [1]: import skrf as rf
	---------------------------------------------------------------------------
	ImportError                               Traceback (most recent call last)
	<ipython-input-1-41c4ee663aa9> in <module>()
	----> 1 import skrf as rf
	\
	ImportError: No module named skrf
	
	
Then installation was unsuccesful. If you need help post to the `mailing list <http://groups.google.com/group/scikit-rf>`_. 


Requirements
------------




Necessary
=============

*    python (>= 2.6 < 3.0 ) http://www.python.org/
*    numpy http://numpy.scipy.org/
*    scipy http://www.scipy.org/
*    matplotlib http://matplotlib.sourceforge.net/


Optional
==========

*    ipython http://ipython.scipy.org/moin/ - for interactive shell
*    pyvisa http://pyvisa.sourceforge.net/pyvisa/ - for instrument control
*    pandas http://pandas.pydata.org/ - for xls spreadsheet export
*    xlrd/xlwt http://www.python-excel.org/ - for xls reading/writing



Debian-Based Linux
======================

For debian-based linux users who dont want to install pythonxy_, here is a one-shot line to install all requirements, ::

	sudo apt-get install python-pyvisa python-numpy python-scipy python-matplotlib ipython python python-setuptools 

Once `setuptools` is installed you can install skrf through `easy_install` ::

	easy_install scikit-rf


.. _pyvisa: http://pyvisa.sourceforge.net/pyvisa/
.. _Python: http://www.python.org/
.. _pythonxy: http://code.google.com/p/pythonxy/


