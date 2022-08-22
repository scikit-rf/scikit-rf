"""
.. module:: skrf.vi

========================================================
virtual instruments (:mod:`skrf.vi`)
========================================================

This module defines "virtual instruments" or interfaces to a number of different
types of instruments used in RF measurements and experiments.

.. automodule:: skrf.vi.vna

Spectrum Analyzers (SAs)
========================

Coming soon!

Antenna Positioners
===================

Coming soon!

Conventions
===========

- `write` means to send a command to an instrument.
- `read` means to read from an instrument.
- `query` means to send a command to an instrument and read its response.
- Interfaces define getters and setters. Getters are simply named the property
  they are getting, and setters are that same name prepended by `set_` (e.g. to
  read the start frequency, the method would be `start_frequency()` and to set
  the start frequency would be `set_start_frequency()`)
- Each method should check if the arguments passed are supported by the
  instrument and raise a `ValueError` otherwise
- If an instrument does not support the associated method, that method should
  raise a `NotImplementedError`


Creating A Driver
=================

Drivers are classes that from the respective instrument type's abstract base
class and define functions specfic to each machine. This syntax can be complex
and vary wildly between devices and manufacturers. The vi module attempts to
present a standard interface for each.

Basically, creating a driver consists of the following:

1. Creating an instrument class that inherits from it's abstract base class
2. Consulting the device's datasheet to find the command for each method in the
   abstract base class

Those wishing to use an instrument that's not supported are encouraged to
consult existing classes to understand the architecture, develop their own
class, and contribute it to scikit-rf.


SCPI Commands
=============

SCPI or Standard Commands for Programmable Instruments is a standard defined by
the IVI Foundation to provide a standard syntax for controlling instruments.
There are many types of instruments that can be controlled with SCPI commands.
The difficult bit is that these commands can vary wildy between manufacturers
and even between devices from the same manufacturer. As an example, consider the
command to change the start frequency of a network anaylyzer to 1GHz.

- Keysight PNA: SENS:FREQ:STAR 1000000000
- Keysight Field Fox: SENS:FREQ:STAR 1000000000
- Rhode and Schwarz: SENS:FREQ:STAR 1000000000

They all look the same right? But what about turning averaging on and off:

- Keysight PNA: SENS:AVER ON
- Keysight FieldFox: ???
- Rhode and Schwarz ZVA: SENS:AVER ON

The PNA and the Rhode and Schwarz share the same command but the FieldFox
doesn't have a command to directly turn averaging on and off. Instead you have
to set the average count to 1 to turn off averaging.

Inconsitencies like this are why we need to create classes for every instrument
we support. Some Instrumentss (like the nanovna) don't even provide a SCPI
interface.

To learn about SCPI, you can read the `IVI website`_, or `Wikipedia`_.

.. _IVI website: http://www.ivifoundation.org/scpi/
.. _Wikipedia: https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments
"""
__test__ = False

from . import vna

all = ["vna"]
