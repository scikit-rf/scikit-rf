.. _virtualinstruments:

Virtual Instruments
********************
.. currentmodule:: skrf.vi
.. contents::

.. warning::

    The vi module is not well written or tested at this point.

The :mod:`~skrf.vi` module holds  classes
for GPIB/VISA instruments that are intricately related to skrf.
Most of the classes were created for the sole purpose of retrieving data 
so that calibration and measurements could be carried out offline with skrf, 
therefore most other instrument capabilities are neglected.

.. note::
    
    To use the virtual instrument classes you must have `pyvisa <http://pyvisa.sourceforge.net/pyvisa/>`_ installed, and a working VISA installation.
    
A list of VNA's that have been are partially supported.


.. hlist::
    :columns: 2
    
    * :class:`~vna.HP8510C`
    * :class:`~vna.HP8720`
    * :class:`~vna.PNA`
    * :class:`~vna.ZVA40`

An example usage of the :class:`~vna.HP8510C` class to retrieve some s-parameter data

.. ipython:: 
    :verbatim:
	
    In [144]: from skrf.vi import vna
    
    In [144]: my_vna = vna.HP8510C(address =16) 
    #if an error is thrown at this point there is most likely a problem with your visa setup
    
    
    In [144]: dut_1 = my_vna.s11
    
    In [144]: dut_2 = my_vna.s21
    
    In [144]: dut_3 = my_vna.two_port
    
Unfortunately, the syntax is different for every VNA class, so the
above example wont directly translate to other VNA's. Re-writing 
all of the VNA classes to follow the same convention is on the 
`TODO list <https://github.com/scikit-rf/scikit-rf/blob/develop/TODO.rst>`_
    
    
