.. _home:

==================================================
scikit-rf *— Object Oriented RF Engineering*
==================================================

**scikit-rf** (aka `skrf`) is an Open Source, BSD-licensed package for RF/Microwave engineering implemented
in the Python programming language. It provides a modern, object-oriented library for network analysis and
calibration which is both flexible and scalable.

Getting Started
---------------

Plot touchstone data on a smith chart, in **3 lines**.

.. code-block:: python

    import skrf as rf
    ntwk = rf.Network('ring slot.s2p')
    ntwk.plot_s_smith()

Install it from `pip <https://pypi.org/project/scikit-rf/>`__ or `conda-forge <https://anaconda.org/conda-forge/scikit-rf>`__,
as described in :doc:`tutorials/Installation`.

.. code-block:: shell

    python -m pip install scikit-rf
    conda install -c conda-forge scikit-rf

Features
--------

.. rst-class:: sk-features

* Microwave Network Operations:

  * Read/Write touchstone (.s2p, s?p) files
  * Arithmetic operations on scattering parameters
  * Cascade/De-embed 2-port networks
  * Frequency and port slicing and concatenation.
  * Connect n-port networks
  * s/z/y/abcd/t - parameter conversion

* Plotting abilities:

  * Rectangular Plots (dB, mag, Phase, group delay)
  * Smith Chart
  * Automated Uncertainty bounds

* Metrology and Offline Calibration:

  * One-Port: SOL, Least Squares, SDDL
  * Two-Port: TRL, Multiline TRL, SOLT, Unknown Thru, 8/16-Term
  * Partial : Enhanced Response, One-Port Two-Path

* Sets of Networks:

  * Statistical properties of NetworkSets
  * Methods to sort and visualize set behavior

* Circuit builder

* Transmission Line Physics:

  * Coax, CPW, Freespace, RectangularWaveguide, DistributedCircuit

Some Examples
-------------

.. nblinkgallery::

    examples/plotting/Modeling RG-58
    examples/metrology/Multiline TRL
    examples/vectorfitting/vectorfitting_ex1_ringslot
    examples/circuit/Wilkinson Power Splitter
    examples/matching/Impedance Matching
    examples/networktheory/Time Domain

Questions or comments? Ask the Community
----------------------------------------

* `LinkedIn scikit-rf Users Group <https://www.linkedin.com/groups/12462155/>`_: User group to ask and answer questions and to received scikit-rf announcements.
* Chat: scikit-rf have channels on both `Slack <https://join.slack.com/t/scikit-rf/shared_invite/zt-d82b62wg-0bdSJjZVhHBKf6687V80Jg>`_ and `Matrix <https://app.element.io/#/room/#scikit-rf:matrix.org>`_
* `Suggestion box <https://docs.google.com/forms/d/e/1FAIpQLSfkSNy8KzoHwdDxYGCmHHiA3GsHgN6zKTFI2NrcUR29U7Xx6A/viewform?usp=sf_link>`_: if you would like to suggest a feature/topic for consideration (optionally anonymous).

If you use scikit-rf, see :doc:`citations_ack/acknowledging`.

.. toctree::
    :caption: Getting started
    :hidden:

    Overview <self>
    tutorials/Installation
    tutorials/Introduction

.. toctree::
    :caption: User guide
    :hidden:

    tutorials/index
    examples/index

.. toctree::
    :caption: Reference
    :hidden:

    api/index
    glossary

.. toctree::
    :caption: Project
    :hidden:

    contributing/index
    citations_ack/index
    license
