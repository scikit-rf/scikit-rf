.. _installation:

****************
Installation
****************


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
anaconda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install scikit-rf is to first install 
`anaconda <http://continuum.io/downloads>`_. Once anaconda is installed you can 
install scikit-rf by entering the following into a terminal::

    conda install -c scikit-rf  scikit-rf
    
If you would like to install the `development` version, use::

    conda install -c https://conda.anaconda.org/scikit-rf/channel/dev scikit-rf

~~~~~~~~~~~~~~~~
pip
~~~~~~~~~~~~~~~~

If you dont want to install anaconda (really, why not?), you can use  `pip`::

    pip install scikit-rf

~~~~~~~~~~~
git
~~~~~~~~~~~

The bleeding-edge development version of **scikit-rf** may be installed using::

    git clone git@github.com:scikit-rf/scikit-rf.git
    cd scikit-rf
    python setup.py install




Other Optional Modules
~~~~~~~~~~~~~~~~~~~~~~

Some features of scikit-rf wont be available untill you install additional
modules. You can install these using conda or pip.

Instrument Control
-----------------------

*   pyvisa http://pyvisa.sourceforge.net/pyvisa/ 
*   python-vxi11 https://pypi.python.org/pypi/python-ivi/
*   python-ivi  https://pypi.python.org/pypi/python-vxi11/ 

Excel file export
---------------------
*   xlrd/xlwt http://www.python-excel.org/ - for xls reading/writing




