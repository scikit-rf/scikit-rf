"""
.. module:: skrf.vi.vna
===============================================
vna (:mod:`skrf.vi.vna`)
===============================================


Provides classes to interact with Vector Network Analyzers (VNAs) from numerous
manufacturers.

VNA Classes
=======================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

   hp.HP8510C
   keysight.FieldFox
   keysight.PNA
   nanovna.NanoVNAv2
   rohde_schwarz.ZVA

The Base Class and Writing an Instrument Driver
===============================================

All VNAs are derived from the :class:`VNA` class. This class should never be
instantiated directly, but instead serves as a means to run certain setup code
depending on the capabilities of the instrument.

    * :ref:`SCPI`
    * :ref:`Channels`
    * :ref:`Property Creator`
    * :ref:`Validators`

When writing a new instrument driver, the following minimum questions should be
considered:
    * Does the instrument use SCPI?
    * Does the instrument support multiple channels?

SCPI
----

For those instruments that use SCPI, default universal SCPI methods are included
by default. However, for those instruments that do not support SPCI, do the
following:

.. code-block:: python

    class VNAWithoutSCPI(VNA):
        _scpi = False
        # ...

Channels
--------

Some instruments support channels and can have multiple measurements with
different frequency/power/if bandwidth/etc settings.

For instruments without channel support, the typical properties (freq_start,
freq_stop, etc) should be defined on the class. For instruments **with** channel
support, the class should include a subclass of :class:`Channel` that defines
properties specific to channel settings in the channel definition while other
instrument properties should be defined in the instrument class. To make this
more clear, consider the following example from the Keysight PNA implementation
(the property creator `VNA.Command` is explained later)

.. code-block:: python

    class PNA(VNA):
        class Channel(vna.Channel):
            freq_start = VNA.Command(
                get_cmd="SENS<self:cnum>:FREQ:STAR?",
                set_cmd="SENS<self:cnum>:FREQ:STAR <arg>",
                doc=\"""Start frequency [Hz]\""",
                validator=FreqValidator()
            )

        def __init__(self, address, backend='@py'):
            # ...

        nports = VNA.Command(
            get_cmd="SYST:CAP:HARD:PORT:COUN?",
            set_cmd=None,
            doc=\"""Number of ports\""",
            validator=IntValidator()
        )

Here, the start frequency is set **per channel** whereas the number of ports is
related to the instrument itself. Instruments with channel support should create a
single channel in `__init__()` using `create_channel`

Property Creator
----------------

Inspired by `PyMeasure <https://github.com/pymeasure/pymeasure>`_, skrf has a property
creator to simplify creating properties that are queried and set with commands.

.. automethod:: VNA.command

For `get_cmd` and `set_cmd`, you can substitute delimiters to be
replaced when the call is made. `self` is always passed, so you can
reference any part of "self" when making the call. Additionally, the
`set_cmd` receives an additional parameter `arg` which is the argument
passed to the setter. These can be used in the get/set strings by using
angle bracket delimiters and referring to the name. Here are some
examples:

Here, we are assuming we are writing a command for an instrument with
channels, and writing a command **for** the internal `Channel` class.
In `get_cmd`, `<self:cnum>` gets the `self.cnum` property of the Channel
class **at runtime**. In `set_cmd`, `<arg>` is replaced by the argument
passed to the setter.

.. code-block:: python

    freq_start = VNA.command(
        get_cmd="CALC<self:cnum>:FREQ:STAR?",
        set_cmd="CALC<self:cnum>:FREQ:STAR <arg>",
    )

And here's an example of calling this function and what the resultant
command strings would be after replacement.

.. code-block:: python

    _ = instr.ch1.freq_start
    # Sends the command CALC1:FREQ:STAR? to the instrument
    instr.ch1.freq_start = 100
    # Sends the command CALC1:FREQ:STAR 100 to the instrument


Validators
----------

Validators are (optionally, but almost always) passed to `VNA.command`. When a property
is get or set, the appropriate validate command is called to convert input to the proper
format expected by the instrument, or convert responses from the instrument to the
proper type.

The current validators are:
    * :class:`skrf.vi.validators.BooleanValidator`
    * :class:`skrf.vi.validators.DelimitedStrValidator`
    * :class:`skrf.vi.validators.DictValidator`
    * :class:`skrf.vi.validators.EnumValidator`
    * :class:`skrf.vi.validators.IntValidator`
    * :class:`skrf.vi.validators.FloatValidator`
    * :class:`skrf.vi.validators.FreqValidator`
    * :class:`skrf.vi.validators.SetValidator`

The documentation for each of these explains more about their functionality, but in essence
when writing a `VNA.command`, consider how the command must be formatted to be sent to the
instrument and what the expected response from the instrument will be and how that can be
transformed to something more useful than, say, a string.

Here's an example of using a validator:

.. code-block:: python

    freq_start = VNA.command(
        get_cmd="CALC:FREQ:STAR?",
        set_cmd="CALC:FREQ:STAR <arg>",
        validator=FreqValidator()
    )

    # ...

    # This will call FreqValidator.validate_input('100 kHz') to
    # transform the string '100 kHz' to '100000'. The full command
    # then becomes:
    # CALC:FREQ:STAR 100000
    instr.ch1.freq_start = '100 Hz'

    # This will send the command CALC:FREQ:STAR? to the instrument
    # then send the response from the instrument to
    # FreqValidator.validate_output() to attempt to transform, as an
    # example, the string '100000' to the int 100_000
    _ = instr.ch1.freq_start
"""

from .vna import VNA, Channel, ValuesFormat  # isort: skip
from . import keysight, nanovna, rohde_schwarz  # isort: skip
