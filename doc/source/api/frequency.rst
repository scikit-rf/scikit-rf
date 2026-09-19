.. automodule:: skrf.frequency

Configuring frequency units
---------------------------

Set ``rf.constants.FREQ_UNIT_DEFAULT`` to choose the unit for newly created
frequency objects and networks, including networks read from Touchstone files::

    import skrf as rf

    rf.constants.FREQ_UNIT_DEFAULT = "GHz"
    implicit = rf.Frequency(1, 2, 2)
    explicit = rf.Frequency(1000, 2000, 2, unit="MHz")

Both objects contain ``[1e9, 2e9]`` in ``f`` (always Hz), report ``unit == "GHz"``,
and expose ``[1, 2]`` through ``f_scaled``. Explicit input units still determine
the meaning of the supplied numbers; the global setting controls their final
representation. ``Frequency.from_f`` follows the same rules.

For a Touchstone file declaring MHz with frequency values 1000 and 2000::

    network = rf.Network("example.s2p")
    # network.f: [1e9, 2e9]
    # network.frequency.unit: "GHz"
    # network.frequency.f_scaled: [1, 2]

File data is interpreted using the file's unit before applying the configured
unit. Scattering parameters, impedance and absolute frequencies are unchanged.
Noise frequency axes use the configured unit as well.

The factory setting is ``None``, which disables automatic unit coercion::

    rf.constants.FREQ_UNIT_DEFAULT = None
    # Explicit constructor units and file units are retained.
    # Frequency(1, 2, 2) still interprets omitted units as Hz.

Changing the setting does not mutate existing objects. Copies and slices retain
the source object's unit; an explicit assignment such as
``frequency.unit = "MHz"`` remains available to change an individual object's
representation. Constructing a new Network from an existing Frequency applies
the current global setting to the network's copy, leaving the source unchanged.

``Network(f=...)`` retains its existing input contract: raw values are in Hz
unless ``f_unit`` specifies another input unit. Its resulting frequency unit
follows the global setting. This differs from omitted ``unit`` in the Frequency
constructor, where the input is interpreted in the configured unit.
