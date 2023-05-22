"""
.. module:: skrf.vi
========================================================
virtual instruments (:mod:`skrf.vi`)
========================================================

This module defines "virtual instruments" or interfaces to a number of different
types of instruments used in RF measurements and experiments.

Creating A Driver
=================

Drivers are classes inherited from the respective instrument type's abstract base
class and define methods specific to each instrument. This syntax can be complex
and vary wildly between devices and manufacturers. The vi module attempts to
present a standard interface for each.

Basically, creating a driver consists of the following:

1. Creating an instrument class that inherits from its abstract base class
2. Consulting the device's datasheet to find the command for each method in the
   abstract base class

Those wishing to use an instrument that's not supported are encouraged to
consult existing classes to understand the architecture, develop their own
class, and contribute it to scikit-rf.

If you choose to do so, please try to use the following standard terms for the
associated meaning

Conventions
-----------

==================== ========================================================
Name                 Meaning
==================== ========================================================
frequency            Frequency settings [:class:`skrf.Frequency`]
freq_start           Start frequency [Hz]
freq_stop            Stop frequency [Hz]
freq_step            Frequency step [Hz]
freq_center          Center frequency [Hz]
freq_span            Frequency span [Hz]
npoints              Number of frequency points
sweep_time           Time taken for a frequency sweep [s]
sweep_type           How frequency points are distributed (linear,log,etc.)
sweep_mode           How the instrument is triggered (continuous,single,etc)
averaging_on         Whether averaging is on or not
averaging_count      The number of measurements combined for an average
averaging_mode       How averages are taken (per point, each sweep, etc)
if_bandwidth         IF bandwidth [Hz]
==================== ========================================================

Validators
----------

.. autosummary::
    :toctree: generated/

    validators.IntValidator
    validators.FloatValidator
    validators.FreqValidator
    validators.EnumValidator
    validators.SetValidator
    validators.DictValidator
    validators.DelimitedStrValidator
    validators.BooleanValidator

"""
from . import validators
