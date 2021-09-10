IO
==

This Package provides functions and objects for input/output.



Reading and writing touchstone files is supported through the
:class:`~touchstone.Touchstone` class, which can be more easily used
through the Network constructor, :func:`~skrf.network.Network.__init__`

The general functions :func:`~skrf.io.general.read` and :func:`~skrf.io.general.write`
can be used to read and write [almost] any skrf object to disk, using the
:mod:`pickle` module. This should only be used for temporary storage,
because pickling is not stable over time, as skrf evolves.


.. automodule:: skrf.io.touchstone

.. automodule:: skrf.io.general

.. automodule:: skrf.io.csv



