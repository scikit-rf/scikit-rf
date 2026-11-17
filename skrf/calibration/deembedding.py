"""
.. module:: skrf.calibration.deembedding

====================================================
deembedding (:mod:`skrf.calibration.deembedding`)
====================================================

De-embedding is the procedure of removing effects of the
test fixture that is often present in the measurement of a device
or circuit. It is based on a lumped element approximation of the
test fixture which needs removal from the raw data, and its
equivalent circuit often needs to be known a-priori. This is often
required since implementation of calibration methods such as
Thru-Reflect-Line (TRL) becomes too expensive for implementation
in on-wafer measurement environments where space is limited, or
insufficiently accurate as in the case of Short-Open-Load-Thru
(SOLT) calibration where the load cannot be manufactured accurately.
De-embedding is often performed as a second step, after a
SOLT, TRL or similar calibration to the end of a known reference
plane, like the probe-tips in on-wafer measurements.

This module provides objects to implement commonly used de-embedding
method in on-wafer applications.
Each de-embedding method inherits from the common abstract base
class :class:`Deembedding`.

Base Class
----------

.. autosummary::
   :toctree: generated/

   Deembedding

De-embedding Methods
--------------------

.. autosummary::
   :toctree: generated/

   OpenShort
   Open
   ShortOpen
   Short
   SplitPi
   SplitTee
   AdmittanceCancel
   ImpedanceCancel
   IEEEP370
   IEEEP370_SE_NZC_2xThru
   IEEEP370_MM_NZC_2xThru
   IEEEP370_SE_ZC_2xThru
   IEEEP370_MM_ZC_2xThru
   Modal_NZC_2xThru

"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy
from numpy import angle, concatenate, conj, exp, flip, imag, ndarray, real, unwrap, zeros
from numpy.fft import fft, fftshift, ifftshift, irfft
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

from .. import mathFunctions as mf
from ..frequency import Frequency
from ..network import Network, concat_ports, overlap_multi, s2t, subnetwork
from ..plotting import figure, subplots

if TYPE_CHECKING:
    from .plotting import Axes, Figure

PortOrderT = Literal["first", "second", "third"]

class Deembedding(ABC):
    """
    Abstract Base Class for all de-embedding objects.

    This class implements the common mechanisms for all de-embedding
    algorithms. Specific calibration algorithms should inherit this
    class and over-ride the methods:
    * :func:`Deembedding.deembed`

    """

    def __init__(self, dummies, name=None, *args, **kwargs):
        r"""
        De-embedding Initializer

        Notes
        -----
        Each de-embedding algorithm may use a different number of
        dummy networks. We check that each of these dummy networks
        have matching frequencies to perform de-embedding.

        It should be known a-priori what the equivalent circuit
        of the parasitic network looks like. The proper de-embedding
        method should then be chosen accordingly.

        Parameters
        ----------
        dummies : list of :class:`~skrf.network.Network` objects
            Network info of all the dummy structures used in a
            given de-embedding algorithm.

        name : string
            Name of this de-embedding instance, like 'open-short-set1'
            This is for convenience of identification.

        \*args, \*\*kwargs : keyword arguments
            stored in self.args and self.kwargs, which may be used
            by sub-classes if needed.
        """

       # ensure all the dummy Networks' frequency's are the same
        for dmyntwk in dummies:
            if dummies[0].frequency != dmyntwk.frequency:
                warnings.warn('Dummy Networks dont have matching frequencies, attempting overlap.', RuntimeWarning,
                              stacklevel=2)
                dummies = overlap_multi(dummies)
                break

        self.frequency = dummies[0].frequency
        self.dummies = dummies
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __str__(self):
        if self.name is None:
            name = ''
        else:
            name = self.name

        output = (f'{self.__class__.__name__} Deembedding: {name}, {self.frequency}, '
                  f'{len(self.dummies)} dummy structures')

        return output

    def __repr_(self):
        return self.__str__()

    @abstractmethod
    def deembed(self, ntwk):
        """
        Apply de-embedding correction to a Network
        """
        pass


class OpenShort(Deembedding):
    """
    Remove open parasitics followed by short parasitics.

    This is a commonly used de-embedding method for on-wafer applications.

    A deembedding object is created with two dummy measurements: `dummy_open`
    and `dummy_short`. When :func:`Deembedding.deembed` is applied,
    Open de-embedding is applied to the short dummy
    because the measurement results for the short dummy contains parallel parasitics.
    Then the Y-parameters of the dummy_open are subtracted from the DUT measurement,
    followed by subtraction of Z-parameters of dummy-short which is previously de-embedded.

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where the series parasitics are closest to device under test,
    followed by the parallel parasitics. For more information, see [1]_

    References
    ------------

    .. [1] M. C. A. M. Koolen, J. A. M. Geelen and M. P. J. G. Versleijen, "An improved
        de-embedding technique for on-wafer high frequency characterization",
        IEEE 1991 Bipolar Circuits and Technology Meeting, pp. 188-191, Sep. 1991.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import OpenShort

    Create network objects for dummy structures and dut

    >>> op = rf.Network('open_ckt.s2p')
    >>> sh = rf.Network('short_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = OpenShort(dummy_open = op, dummy_short = sh, name = 'test_openshort')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)

    """

    def __init__(self, dummy_open, dummy_short, name=None, *args, **kwargs):
        """
        Open-Short De-embedding Initializer

        Parameters
        -----------

        dummy_open : :class:`~skrf.network.Network` object
            Measurement of the dummy open structure

        dummy_short : :class:`~skrf.network.Network` object
            Measurement of the dummy short structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.open = dummy_open.copy()
        self.short = dummy_short.copy()
        dummies = [self.open, self.short]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding


        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.open.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning,
                          stacklevel=2)
            caled, op, sh = overlap_multi([ntwk, self.open, self.short])
        else:
            caled, op, sh = ntwk.copy(), self.open, self.short

        # remove parallel parasitics from the short dummy
        deembeded_short = sh.copy()
        deembeded_short.y = sh.y - op.y
        # remove parallel parasitics from the dut
        caled.y = caled.y - op.y
        # remove series parasitics from the dut
        caled.z = caled.z - deembeded_short.z

        return caled


class Open(Deembedding):
    """
    Remove open parasitics only.

    A deembedding object is created with just one open dummy measurement,
    `dummy_open`. When :func:`Deembedding.deembed` is applied, the
    Y-parameters of the open dummy are subtracted from the DUT measurement,

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where the series parasitics are assumed to be negligible,
    but parallel parasitics are unwanted.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import Open

    Create network objects for dummy structure and dut

    >>> op = rf.Network('open_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = Open(dummy_open = op, name = 'test_open')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)
    """

    def __init__(self, dummy_open, name=None, *args, **kwargs):
        """
        Open De-embedding Initializer

        Parameters
        -----------

        dummy_open : :class:`~skrf.network.Network` object
            Measurement of the dummy open structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.open = dummy_open.copy()
        dummies = [self.open]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.open.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning,
                           stacklevel=2)
            ntwk, op = overlap_multi([ntwk, self.open])
        else:
            op = self.open

        caled = ntwk.copy()
        # remove open parasitics
        caled.y = ntwk.y - op.y

        return caled


class ShortOpen(Deembedding):
    """
    Remove short parasitics followed by open parasitics.

    A deembedding object is created with two dummy measurements: `dummy_open`
    and `dummy_short`. When :func:`Deembedding.deembed` is applied,
    short de-embedding is applied to the open dummy
    because the measurement results for the open dummy contains series parasitics.
    the Z-parameters of the dummy_short are subtracted from the DUT measurement,
    followed by subtraction of Y-parameters of dummy_open which is previously de-embedded.

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where the parallel parasitics are closest to device under test,
    followed by the series parasitics.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import ShortOpen

    Create network objects for dummy structures and dut

    >>> op = rf.Network('open_ckt.s2p')
    >>> sh = rf.Network('short_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = ShortOpen(dummy_short = sh, dummy_open = op, name = 'test_shortopen')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)

    """

    def __init__(self, dummy_short, dummy_open, name=None, *args, **kwargs):
        """
        Short-Open De-embedding Initializer

        Parameters
        -----------

        dummy_short : :class:`~skrf.network.Network` object
            Measurement of the dummy short structure

        dummy_open : :class:`~skrf.network.Network` object
            Measurement of the dummy open structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.open = dummy_open.copy()
        self.short = dummy_short.copy()
        dummies = [self.open, self.short]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.open.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning,
                          stacklevel=2)
            ntwk, op, sh = overlap_multi([ntwk, self.open, self.short])
        else:
            op, sh = self.open, self.short

        caled = ntwk.copy()
        # remove series parasitics from the open dummy
        deembeded_open = op.copy()
        deembeded_open.z = op.z - sh.z
        # remove parallel parasitics from the dut
        caled.z = ntwk.z - sh.z
        # remove series parasitics from the dut
        caled.y = caled.y - deembeded_open.y

        return caled


class Short(Deembedding):
    """
    Remove short parasitics only.

    This is a useful method to remove pad contact resistances from measurement.

    A deembedding object is created with just one dummy measurement: `dummy_short`.
    When :func:`Deembedding.deembed` is applied, the
    Z-parameters of the dummy_short are subtracted from the DUT measurement,

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where only series parasitics are to be removed while retaining all others.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import Short

    Create network objects for dummy structures and dut

    >>> sh = rf.Network('short_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = Short(dummy_short = sh, name = 'test_short')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)

    """

    def __init__(self, dummy_short, name=None, *args, **kwargs):
        """
        Short De-embedding Initializer

        Parameters
        -----------

        dummy_short : :class:`~skrf.network.Network` object
            Measurement of the dummy short structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.short = dummy_short.copy()
        dummies = [self.short]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.short.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning,
                           stacklevel=2)
            ntwk, sh = overlap_multi([ntwk, self.short])
        else:
            sh = self.short

        caled = ntwk.copy()
        # remove short parasitics
        caled.z = ntwk.z - sh.z

        return caled


class SplitPi(Deembedding):
    """
    Remove shunt and series parasitics assuming pi-type embedding network.

    A deembedding object is created with just one thru dummy measurement `dummy_thru`.
    The thru dummy is, for example, a direct cascaded connection of the left and right test pads.

    When :func:`Deembedding.deembed` is applied,
    the shunt admittance and series impedance of the thru dummy are removed.

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where the series parasitics are closest to device under test,
    followed by the shunt parasitics. For more information, see [2]_

    References
    ------------
    ..  [2] L. Nan, K. Mouthaan, Y.-Z. Xiong, J. Shi, S. C. Rustagi, and B.-L. Ooi,
        “Experimental Characterization of the Effect of Metal Dummy Fills on Spiral Inductors,”
        in 2007 IEEE Radio Frequency Integrated Circuits (RFIC) Symposium, Jun. 2007, pp. 307–310.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import SplitPi

    Create network objects for dummy structure and dut

    >>> th = rf.Network('thru_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = SplitPi(dummy_thru = th, name = 'test_thru')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)
    """

    def __init__(self, dummy_thru, name=None, *args, **kwargs):
        """
        SplitPi De-embedding Initializer

        Parameters
        -----------
        dummy_thru : :class:`~skrf.network.Network` object
            Measurement of the dummy thru structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`
        """
        self.thru = dummy_thru.copy()
        dummies = [self.thru]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding
        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.thru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, thru = overlap_multi([ntwk, self.thru])
        else:
            thru = self.thru

        left = thru.copy()
        left_y = left.y
        left_y[:,0,0] = (thru.y[:,0,0] - thru.y[:,1,0] + thru.y[:,1,1] - thru.y[:,0,1]) / 2
        left_y[:,0,1] = thru.y[:,1,0] + thru.y[:,0,1]
        left_y[:,1,0] = thru.y[:,1,0] + thru.y[:,0,1]
        left_y[:,1,1] = - thru.y[:,1,0] - thru.y[:,0,1]
        left.y = left_y
        right = left.flipped()
        caled = left.inv ** ntwk ** right.inv

        return caled


class SplitTee(Deembedding):
    """
    Remove series and shunt parasitics assuming tee-type embedding network.

    A deembedding object is created with just one thru dummy measurement `dummy_thru`.
    The thru dummy is, for example, a direct cascaded connection of the left and right test pads.

    When :func:`Deembedding.deembed` is applied,
    the shunt admittance and series impedance of the thru dummy are removed.

    This method is applicable only when there is a-priori knowledge of the
    equivalent circuit model of the parasitic network to be de-embedded,
    where the shunt parasitics are closest to device under test,
    followed by the series parasitics. For more information, see [3]_

    References
    ------------
    ..  [3] M. J. Kobrinsky, S. Chakravarty, D. Jiao, M. C. Harmes, S. List, and M. Mazumder,
        “Experimental validation of crosstalk simulations for on-chip interconnects using S-parameters,”
        IEEE Transactions on Advanced Packaging, vol. 28, no. 1, pp. 57–62, Feb. 2005.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import SplitTee

    Create network objects for dummy structure and dut

    >>> th = rf.Network('thru_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = SplitTee(dummy_thru = th, name = 'test_thru')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)
    """

    def __init__(self, dummy_thru, name=None, *args, **kwargs):
        """
        SplitTee De-embedding Initializer

        Parameters
        -----------
        dummy_thru : :class:`~skrf.network.Network` object
            Measurement of the dummy thru structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`
        """
        self.thru = dummy_thru.copy()
        dummies = [self.thru]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding
        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.thru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, thru = overlap_multi([ntwk, self.thru])
        else:
            thru = self.thru

        left = thru.copy()
        left_z = left.z
        left_z[:,0,0] = (thru.z[:,0,0] + thru.z[:,1,0] + thru.z[:,1,1] + thru.z[:,0,1]) / 2
        left_z[:,0,1] = thru.z[:,1,0] + thru.z[:,0,1]
        left_z[:,1,0] = thru.z[:,1,0] + thru.z[:,0,1]
        left_z[:,1,1] = thru.z[:,1,0] + thru.z[:,0,1]
        left.z = left_z
        right = left.flipped()
        caled = left.inv ** ntwk ** right.inv

        return caled


class AdmittanceCancel(Deembedding):
    """
    Cancel shunt admittance by swapping (a.k.a Mangan's method).
    A deembedding object is created with just one thru dummy measurement `dummy_thru`.
    The thru dummy is, for example, a direct cascaded connection of the left and right test pads.

    When :func:`Deembedding.deembed` is applied,
    the shunt admittance of the thru dummy are canceled,
    from the DUT measurement by left-right mirroring operation.

    This method is applicable to only symmetric (i.e. S11=S22 and S12=S21) 2-port DUTs,
    but suitable for the characterization of transmission lines at mmW frequencies.
    For more information, see [4]_

    References
    ------------
    ..  [4] A. M. Mangan, S. P. Voinigescu, Ming-Ta Yang, and M. Tazlauanu,
        “De-embedding transmission line measurements for accurate modeling of IC designs,”
        IEEE Trans. Electron Devices, vol. 53, no. 2, pp. 235–241, Feb. 2006.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import AdmittanceCancel

    Create network objects for dummy structure and dut

    >>> th = rf.Network('thru_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = AdmittanceCancel(dummy_thru = th, name = 'test_thru')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)
    """

    def __init__(self, dummy_thru, name=None, *args, **kwargs):
        """
        AdmittanceCancel De-embedding Initializer

        Parameters
        -----------

        dummy_thru : :class:`~skrf.network.Network` object
            Measurement of the dummy thru structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.thru = dummy_thru.copy()
        dummies = [self.thru]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------

        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------

        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding
        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.thru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, thru = overlap_multi([ntwk, self.thru])
        else:
            thru = self.thru

        caled = ntwk.copy()
        h = ntwk ** thru.inv
        h_ = h.flipped()
        caled.y = (h.y + h_.y) / 2

        return caled


class ImpedanceCancel(Deembedding):
    """
    Cancel series impedance by swapping.

    A deembedding object is created with just one thru dummy measurement `dummy_thru`.
    The thru dummy is, for example, a direct cascaded connection of the left and right test pads.

    When :func:`Deembedding.deembed` is applied,
    the series impedance of the thru dummy are canceled,
    from the DUT measurement by left-right mirroring operation.

    This method is applicable to only symmetric (i.e. S11=S22 and S12=S21) 2-port DUTs,
    but suitable for the characterization of transmission lines at mmW frequencies.
    For more information, see [5]_

    References
    ------------
    ..  [5] S. Amakawa, K. Katayama, K. Takano, T. Yoshida, and M. Fujishima,
        “Comparative analysis of on-chip transmission line de-embedding techniques,”
        in 2015 IEEE International Symposium on Radio-Frequency Integration Technology,
        Sendai, Japan, Aug. 2015, pp. 91–93.


    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import ImpedanceCancel

    Create network objects for dummy structure and dut

    >>> th = rf.Network('thru_ckt.s2p')
    >>> dut = rf.Network('full_ckt.s2p')

    Create de-embedding object

    >>> dm = ImpedanceCancel(dummy_thru = th, name = 'test_thru')

    Remove parasitics to get the actual device network

    >>> realdut = dm.deembed(dut)
    """

    def __init__(self, dummy_thru, name=None, *args, **kwargs):
        """
        ImpedanceCancel De-embedding Initializer

        Parameters
        -----------

        dummy_thru : :class:`~skrf.network.Network` object
            Measurement of the dummy thru structure

        name : string
            Optional name of de-embedding object

        args, kwargs:
            Passed to :func:`Deembedding.__init__`

        See Also
        ---------

        :func:`Deembedding.__init__`
        """
        self.thru = dummy_thru.copy()
        dummies = [self.thru]

        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    def deembed(self, ntwk):
        """
        Perform the de-embedding calculation

        Parameters
        ----------

        ntwk : :class:`~skrf.network.Network` object
            Network data of device measurement from which
            parasitics needs to be removed via de-embedding

        Returns
        -------

        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding
        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.thru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, thru = overlap_multi([ntwk, self.thru])
        else:
            thru = self.thru

        caled = ntwk.copy()
        h = ntwk ** thru.inv
        h_ = h.flipped()
        caled.z = (h.z + h_.z) / 2

        return caled


class IEEEP370(Deembedding):
    """
    Abstract Base Class for all IEEEP370 de-embedding class.

    This class implements the common mechanisms for all IEEEP370 de-embedding
    algorithms. Specific algorithms should inherit this
    class and override the methods:
    * :func:`IEEEP370.deembed`
    * :func:`IEEEP370.split2xthru`

    Based on [ElSA20]_, [I3E3701]_, [I3E3702]_, [I3E3703]_, [I3E3704]_,
    and [I3E3705]_.

    References
    ----------
    .. [ElSA20] Ellison J, Smith SB, Agili S., "Using a 2x-thru standard to achieve
        accurate de-embedding of measurements", Microwave Optical Technology
        Letter, 2020, https://doi.org/10.1002/mop.32098
    .. [I3E3701] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP3702xThru.m,
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    .. [I3E3702] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370mmZc2xthru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    .. [I3E3703] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370Zc2xThru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    .. [I3E3704] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370mmZc2xthru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    .. [I3E3705] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG3/qualityCheckFrequencyDomain.m
       commit 8b8f3a3b5e41aeb4ab16110bbfb683ec52e70206
    """
    def __init__(self, dummies: Sequence[Network], name: str = None,
                 *args, **kwargs) -> None:
        r"""
        IEEEP370 de-embedding Initializer.

        Parameters
        ----------
        dummies : list of :class:`~skrf.network.Network` objects
            Network info of all the dummy structures used in a
            given de-embedding algorithm.

        name : string
            Name of this de-embedding instance, like 'open-short-set1'
            This is for convenience of identification.

        \*args, \*\*kwargs : keyword arguments
            stored in self.args and self.kwargs, which may be used
            by sub-classes if needed.
        """
        Deembedding.__init__(self, dummies, name, *args, **kwargs)

    @abstractmethod
    def deembed(self, ntwk: Network) -> Network:
        """
        Apply de-embedding correction to a Network
        """
        pass

    @abstractmethod
    def split2xthru(self):
        """
        Determine fixtures models
        """
        pass

    @staticmethod
    def extrapolate_to_dc(ntwk: Network) -> Network:
        """
        Extrapolate the network to DC using IEEE370 NZC algorithm.
        This is useful to compare the fixtures and deembedded networks
        to the input data in the same conditions used by NZC algorithm.
        If the network already have a DC point, it will be replaced.

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            Network to be extrapolated to DC

        Returns
        -------
        ntwk_dc : :class:`~skrf.network.Network` object
            Network with DC point

        """
        name = ntwk.name
        s = ntwk.s
        f = ntwk.frequency.f
        port_modes = ntwk.port_modes
        # check for already existing DC point
        if(f[0] == 0):
            warnings.warn(
                "Existing DC point is replaced by extrapolated value.",
                RuntimeWarning, stacklevel=2
                )
            f = f[1:]
            s = s[1:]
        # check for bad frequency vector
        df = f[1] - f[0]
        tol = 0.1 # allow a tolerance of 0.1 from delta-f to starting f (prevent non-issues from precision)
        if(np.abs(f[0] - df) > tol):
            warnings.warn(
               """Non-uniform frequency vector detected. Consider interpolation.""",
               RuntimeWarning, stacklevel=2
               )
        n_ports = ntwk.number_of_ports
        z0 = ntwk.z0[0]
        n = len(f)
        snew = zeros((n + 1, n_ports, n_ports), dtype = complex)
        snew[1:,:,:] = s
        for i in range(n_ports):
            for j in range(n_ports):
                if i == j:
                    snew[0, i, j] = IEEEP370.DC(s[:, i, j], f)
                else:
                    snew[0, i, j] = IEEEP370.dc_interp(s[:, i, j], f)

        f = concatenate(([0], f))
        ntwk_dc = Network(frequency = Frequency.from_f(f, 'Hz'), s = snew,
                       z0 = z0, name = name)
        ntwk_dc.port_modes = port_modes
        return ntwk_dc

    @staticmethod
    def dc_interp(s: ndarray, f: ndarray) -> float:
        """
        enforces symmetric upon the first 10 points and interpolates the DC
        point.
        """
        sp = s[0:9]
        fp = f[0:9]

        snp = concatenate((conj(flip(sp)), sp))
        fnp = concatenate((-1*flip(fp), fp))
        # mhuser : used cubic instead spline (not implemented)
        snew = scipy.interpolate.interp1d(fnp, snp, axis=0, kind = 'cubic')
        return real(snew(0))

    @staticmethod
    def COM_receiver_noise_filter(f: ndarray, fr: float) -> ndarray:
        """
        receiver filter in COM defined by eq 93A-20
        As defined in 802.3-2022 - IEEE Standard for Ethernet annex 93A
        """
        fdfr = f / fr
        # eq 93A-20
        return 1 / (1 - 3.414214 * fdfr**2 + fdfr**4 + 1j*2.613126*(fdfr - fdfr**3))

    @staticmethod
    def makeStep(impulse: ndarray) -> ndarray:
        """
        Make a time-domain step response from an impulse response.
        """
        #mhuser : no need to call step function here, cumsum will be enough and efficient
        #step = np.convolve(np.ones((len(impulse))), impulse)
        #return step[0:len(impulse)]
        return np.cumsum(impulse, axis=0)

    @staticmethod
    def DC(s: ndarray, f: ndarray, allowedError: float = 1e-12) -> float:
        """
        Advanced reflective DC point extrapolation.
        """
        DCpoint = 0.002 # seed for the algorithm
        err = 1 # error seed
        cnt = 0
        df = f[1] - f[0]
        n = len(f)
        t = np.linspace(-1/df,1/df,n*2+1)
        ts = np.argmin(np.abs(t - (-3e-9)))
        Hr = IEEEP370.COM_receiver_noise_filter(f, f[-1]/2)
        while(err > allowedError):
            h1 = IEEEP370.makeStep(
                fftshift(irfft(concatenate(([DCpoint], Hr * s)), axis=0), axes=0))
            h2 = IEEEP370.makeStep(
                fftshift(irfft(concatenate(([DCpoint + 0.001], Hr * s)), axis=0), axes=0))
            m = (h2[ts] - h1[ts]) / 0.001
            b = h1[ts] - m * DCpoint
            DCpoint = (0 - b) / m
            err = np.abs(h1[ts] - 0)
            cnt += 1
        return DCpoint

    @staticmethod
    def thru(ntwk: Network) -> Network:
        """
        Create a perfect thru

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
               Network from which copy frequency, z0 and other parameters.
               The S-parameters will be replaced by zero-length matched lossless
               thru.

        Returns
        -------
        out : :class:`~skrf.network.Network` object
              Network of the perfect thru

        """
        out = ntwk.copy()
        out.s[:, 0, 0] = 0
        out.s[:, 1, 0] = 1
        out.s[:, 0, 1] = 1
        out.s[:, 1, 1] = 0
        return out

    @staticmethod
    def add_dc(ntwk: Network) -> Network:
        """
        Extrapolate a network to DC using interpolation for all S-parameters.

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
               Network to be extrapolated to DC

        Returns
        -------
        out : :class:`~skrf.network.Network` object
              Network with DC point

        """
        s = ntwk.s
        f = ntwk.frequency.f
        z0 = ntwk.z0[0]
        n = len(f)
        snew = zeros((n + 1, 2,2), dtype = complex)
        snew[1:,:,:] = s
        snew[0, 0, 0] = IEEEP370.dc_interp(s[:, 0, 0], f)
        snew[0, 0, 1] = IEEEP370.dc_interp(s[:, 0, 1], f)
        snew[0, 1, 0] = IEEEP370.dc_interp(s[:, 1, 0], f)
        snew[0, 1, 1] = IEEEP370.dc_interp(s[:, 1, 1], f)

        f = concatenate(([0], f))
        return Network(frequency = Frequency.from_f(f, 'Hz'), s = snew, z0 = z0)

    @staticmethod
    def getz(s: ndarray, f: ndarray, z0: float) -> ndarray:
        """
        Compute step response to get the time-domain impedance from S-parameters.
        The S-parameters are DC extrapolated first.

        Parameters
        ----------
        s : :array-like
            1-Port S-parameters array
        f : :array-like
            Frequency array for DC extrapolation
        z0: :array-like
            Reference impedance

        Returns
        -------
        z : :array-like
            Time-domain impedance step response

        """
        DC11 = IEEEP370.DC(s, f, 1e-10)
        t112x = irfft(concatenate(([DC11], s)))
        #get the step response of t112x. Shift is needed for makeStep to
        #work properly.
        t112xStep = IEEEP370.makeStep(fftshift(t112x))
        #construct the transmission line
        z = -z0 * (t112xStep + 1) / (t112xStep - 1)
        z = ifftshift(z) #impedance. Shift again to get the first point first.
        return z

    @staticmethod
    def makeTL(zline: float, z0: float, gamma: ndarray, l: float) -> ndarray:
        """
        Compute the S-parameters of a transmission line.

        Parameters
        ----------
        zline : :number
                Characteristic impedance
        z0    : :number
                Port impedance to renormalize into
        gamma : :array-like
                Frequency-dependent propagation constant
        l    : :number
                Length in the same length unit as gamma

        Returns
        -------
        TL : :array-like
             S_Parameters of the transmission line
        """
        # todo: use DefinedGammaZ0 media instead
        n = len(gamma)
        TL = np.zeros((n, 2, 2), dtype = complex)
        TL[:, 0, 0] = (((zline**2 - z0**2) * np.sinh(gamma * l))
                       / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l)))
        TL[:, 1, 0] = (2 * z0 * zline) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        TL[:, 0, 1] = (2 * z0 * zline) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        TL[:, 1, 1] = (((zline**2 - z0**2) * np.sinh(gamma * l))
                       / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l)))
        return TL

    @staticmethod
    def NRP(ntwk: Network, TD: ndarray = None, port: int = None) -> (Network, ndarray):
        """
        Enforce the Nyquist Rate Point.
        Force the length of the transmissive network to be an integer multiple
        of the wavelength at the highest frequency.
        If required, a proper delay is added to meet this condition.
        The function can also be used to remove the delay.

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
               Network to be extrapolated to DC
        TD   : :array-like
               If None, the delay will be computed and added.
               Else, the delay will be removed (to reset the original length).
               (default: None)
        port: :Number
              Specify to apply NRP only on a single port of the network (default: None)

        Returns
        -------
        TL : :array-like
             S_Parameters of the transmission line
        """
        p = ntwk.s
        f = ntwk.frequency.f
        n = len(f)
        X = ntwk.nports
        fend = f[-1]
        if TD is None:
            TD = np.zeros(X)
            for i in range(X):
                theta0 = angle(p[-1, i, i])
                if theta0 < -np.pi/2:
                    theta = -np.pi - theta0
                elif theta0 > np.pi/2:
                    theta = np.pi - theta0
                else:
                    theta = -theta0

                TD[i] = -theta / (2 * np.pi * fend)
                pd = np.zeros((n, X, X), dtype = complex)
                delay = exp(-1j * 2. * np.pi * f * TD[i] / 2.)
                if i == 0:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = ntwk.copy()
                    spd.s = pd
                    out = spd ** ntwk
                elif i < X//2:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = ntwk.copy()
                    spd.s = pd
                    out = spd ** out
                else:
                    pd[:, i - X//2, i] = delay
                    pd[:, i, i - X//2] = delay
                    spd = ntwk.copy()
                    spd.s = pd
                    out = out ** spd
        else:
            pd = np.zeros((n, X, X), dtype = complex)
            if port is not None:
                i = port
                delay = exp(1j * 2. * np.pi * f * TD[i] / 2.)
                if i < X//2:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = ntwk.copy()
                    spd.s = pd
                    out = spd ** ntwk
                else:
                    pd[:, i - X//2, i] = delay
                    pd[:, i, i - X//2] = delay
                    spd = ntwk.copy()
                    spd.s = pd
                    out = ntwk ** spd
            else:
                for i in range(X):
                    delay = exp(1j * 2. * np.pi * f * TD[i] / 2)
                    if i == 0:
                        pd[:, i + X//2, i] = delay
                        pd[:, i, i + X//2] = delay
                        spd = ntwk.copy()
                        spd.s = pd
                        out = spd ** ntwk
                    elif i < X//2:
                        pd[:, i + X//2, i] = delay
                        pd[:, i, i + X//2] = delay
                        spd = ntwk.copy()
                        spd.s = pd
                        out = spd ** out
                    else:
                        pd[:, i - X//2, i] = delay
                        pd[:, i, i - X//2] = delay
                        spd = ntwk.copy()
                        spd.s = pd
                        out = out ** spd
        return out, TD

    @staticmethod
    def shiftOnePort(ntwk: Network, N: int, port: int) -> Network:
        """
        Shift one port of the network of N samples in time-domain.
        This is achieved by cascading a delay.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Network to be shifted
        N   : :number
              Number of point to shift
        port: :Number
              Port to be shifted

        Returns
        -------
        out : :class:`~skrf.network.Network` object
              Shifted network
        """
        f = ntwk.frequency.f
        n = len(f)
        X = ntwk.nports
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        delay = exp(-N * 1j * Omega/2)
        pd = np.zeros((n, 2, 2), dtype = complex)
        if port < X//2:
            pd[:, port, port + X//2] = delay
            pd[:, port + X//2, port] = delay
            spd = ntwk.copy()
            spd.s = pd
            out = spd ** ntwk
        else:
            pd[:, port, port - X//2] = delay
            pd[:, port - X//2, port] = delay
            spd = ntwk.copy()
            spd.s = pd
            out = ntwk ** spd
        return out

    @staticmethod
    def shiftNPoints(ntwk: Network, N: int) -> Network:
        """
        Shift the whole network of N samples in time-domain.
        This is achieved by cascading a delay.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Network to be shifted
        N   : :number
              Number of point to shift

        Returns
        -------
        out : :class:`~skrf.network.Network` object
              Shifted network
        """
        f = ntwk.frequency.f
        n = len(f)
        X = ntwk.nports
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        delay = exp(-N * 1j * Omega/2)
        pd = np.zeros((n, 2, 2), dtype = complex)
        for port in range(X):
            if port < X//2:
                pd[:, port, port + X//2] = delay
                pd[:, port + X//2, port] = delay
                spd = ntwk.copy()
                spd.s = pd
                p = spd ** ntwk
            else:
                pd[:, port, port - X//2] = delay
                pd[:, port - X//2, port] = delay
                spd = ntwk.copy()
                spd.s = pd
                out = p ** spd
        return out

    @staticmethod
    def peelNPointsLossless(ntwk: Network, N: int, z0: float) -> Network:
        """
        Peel N points of the network on both side and return the corresponding
        error boxes.
        This is done in a lossless way without determination of the propagation
        constant gamma.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Network to be peeled
        N     : :number
                Number of points to peel
        z0    : :number
                Reference impedance
        gamma : :array-like
                Frequency-dependent propagation constant

        Returns
        -------
        out : :class:`~skrf.network.Network` object
              Peeled network
        out : :class:`~skrf.network.Network` object
              Error box side port 1
        out : :class:`~skrf.network.Network` object
              Error box side port 2
        """
        f = ntwk.frequency.f
        n = len(f)
        out = ntwk.copy()
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        betal = 1j * Omega/2
        for i in range(N):
            p = out.s
            #calculate impedance
            zline1 = IEEEP370.getz(p[:, 0, 0], f, z0)[0]
            zline2 = IEEEP370.getz(p[:, 1, 1], f, z0)[0]
            #this is the transmission line to be removed
            TL1 = IEEEP370.makeTL(zline1, z0, betal, 1)
            TL2 = IEEEP370.makeTL(zline2, z0, betal, 1)
            sTL1 = ntwk.copy()
            sTL1.s = TL1
            sTL2 = ntwk.copy()
            sTL2.s = TL2
            #remove the errorboxes
            # no need to flip sTL2 because it is symmetrical
            out = sTL1.inv ** out ** sTL2.inv
            #capture the errorboxes from side 1 and 2
            if i == 0:
                eb1 = sTL1.copy()
                eb2 = sTL2.copy()
            else:
                eb1 = eb1 ** sTL1
                eb2 = sTL2 ** eb2

        return out, eb1, eb2

class IEEEP370_FER:
    """
    IEEE 370 checking for fixture electrical requirements (FER) in the
    frequency and in the time domains.

    Based on [IEEE370]_.

    References
    ----------
    .. [IEEE370] IEEE Standard for Electrical Characterization of Printed
    Circuit Board and Related Interconnects at Frequencies up to 50 GHz",
    IEEE 370-2020.
    """
    def plot_constant_limit(self, frequency: Frequency, value: float, ax: Axes,
                            **kwargs) -> None:
        """
        Plot a constant limit line.
        """
        ax.plot([frequency.f[0], frequency.f[-1]], [value, value], **kwargs)

    def plot_relative_limit(self, x: ndarray, y: ndarray, value: float, ax: Axes,
                            **kwargs) -> None:
        """
        Plot positive and negative relative limit line around a reference trace.
        """
        ax.plot(x, y * (1.0 + value), **kwargs)
        kwargs.pop('label', None)
        ax.plot(x, y * (1.0 - value), label = '_nolabel_', **kwargs)

    def plot_limit_fer1(self, frequency: Frequency, ax: Axes) -> None:
        """
        Plot fer 1 limit lines.
        """
        self.plot_constant_limit(frequency, -10, ax, color = 'g',
                            linestyle = 'dashed', label = 'Minimum A')
        self.plot_constant_limit(frequency, -15, ax, color = 'r',
                            linestyle = 'dashed', label = 'Minimum B, C')

    def plot_limit_fer2(self, frequency: Frequency, ax: Axes) -> None:
        """
        Plot fer 2 limit lines.
        """
        self.plot_constant_limit(frequency, -20, ax, color = 'g',
                            linestyle = 'dashed', label = 'Maximum A')
        self.plot_constant_limit(frequency, -10, ax, color = 'b',
                            linestyle = 'dashed', label = 'Maximum B')
        self.plot_constant_limit(frequency, -6, ax, color = 'r',
                            linestyle = 'dashed', label = 'Maximum C')

    def plot_limit_fer3(self, frequency: Frequency, ax: Axes) -> None:
        """
        Plot fer 3 limit lines.
        """
        self.plot_constant_limit(frequency, 5, ax, color = 'g',
                            linestyle = 'dashed', label = 'Minimum A')
        self.plot_constant_limit(frequency, 0, ax, color = 'r',
                            linestyle = 'dashed', label = 'Minimum B, C')

    def plot_limit_fer5(self, x: ndarray, y: ndarray, ax: Axes) -> None:
        """
        Plot fer 5 limit lines.
        """
        self.plot_relative_limit(x, y, 0.025, ax, linestyle = 'dashed', color = 'g',
                                     label = 'Limit A ±2.5%')
        self.plot_relative_limit(x, y, 0.05, ax, linestyle = 'dashed', color = 'b',
                                     label = 'Limit B ±5%')
        self.plot_relative_limit(x, y, 0.1, ax, linestyle = 'dashed', color = 'r',
                                     label = 'Limit C ±10%')

    def plot_limit_fer6(self, frequency: Frequency, ax: Axes) -> None:
        """
        Plot fer 6 limit lines.
        """
        self.plot_constant_limit(frequency, -15, ax, color = 'r',
                            linestyle = 'dashed', label = 'Maximum A, B, C')



    def plot_fd_se_fer(self, s2xthru: Network, fig: Figure = None) -> Figure:
        """
        Plot fixture electrical requirements (FER) for s values
        """
        if fig is None:
            fig = figure(figsize=(8, 8))

        fig.suptitle('Fixture electrical requirements (FER)')

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('FER1 2x-Thru IL')
        s2xthru.plot_s_db(1, 0, ax = ax, color = '0.5')
        s2xthru.plot_s_db(0, 1, ax = ax, color = 'k')
        self.plot_limit_fer1(s2xthru.frequency, ax)
        ax.legend(loc = 'lower left')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('FER2 2x-Thru RL')
        s2xthru.plot_s_db(0, 0, ax = ax, color = '0.5')
        s2xthru.plot_s_db(1, 1, ax = ax, color = 'k')
        self.plot_limit_fer2(s2xthru.frequency, ax)
        ax.legend(loc = 'lower left')

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title('FER3 2x-Thru IL - RL')
        s1 = s2xthru.s_db[:, 1, 0] - s2xthru.s_db[:, 0, 0]
        s2 = s2xthru.s_db[:, 0, 1] - s2xthru.s_db[:, 1, 1]
        ax.plot(s2xthru.frequency.f, s1, color = '0.5', label = 'S21 - S11')
        ax.plot(s2xthru.frequency.f, s2, color = 'k', label = 'S21 - S22')
        self.plot_limit_fer3(s2xthru.frequency, ax)
        ax.set_xlabel(f'Frequency ({s2xthru.frequency.unit})')
        ax.set_ylabel('Magnitude (dB)')
        ax.legend(loc = 'upper right')

        fig.tight_layout()
        return fig

    def plot_fd_mm_fer(self, s2xthru: Network, fig: Figure = None) -> Figure:
        """
        Plot fixture electrical requirements (FER) for s values
        """
        if fig is None:
            fig = figure(figsize=(8, 8))

        fig.suptitle('Fixture electrical requirements (FER)')

        mm_2xthru = s2xthru.copy()
        mm_2xthru.se2gmm(p=2)

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('FER1 2x-Thru IL')
        mm_2xthru.plot_s_db(1, 0, ax = ax, color = '0.5')
        mm_2xthru.plot_s_db(0, 1, ax = ax, color = 'k')
        self.plot_limit_fer1(mm_2xthru.frequency, ax)
        ax.legend(loc = 'lower left')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('FER2 2x-Thru RL')
        mm_2xthru.plot_s_db(0, 0, ax = ax, color = '0.5')
        mm_2xthru.plot_s_db(1, 1, ax = ax, color = 'k')
        self.plot_limit_fer2(mm_2xthru.frequency, ax)
        ax.legend(loc = 'lower left')

        ax = fig.add_subplot(2, 2, 3)
        ax.set_title('FER3 2x-Thru IL - RL')
        s1 = mm_2xthru.s_db[:, 1, 0] - mm_2xthru.s_db[:, 0, 0]
        s2 = mm_2xthru.s_db[:, 0, 1] - mm_2xthru.s_db[:, 1, 1]
        ax.plot(mm_2xthru.frequency.f, s1, color = '0.5', label = 'S21 - S11')
        ax.plot(mm_2xthru.frequency.f, s2, color = 'k', label = 'S21 - S22')
        self.plot_limit_fer3(mm_2xthru.frequency, ax)
        ax.set_xlabel(f'Frequency ({mm_2xthru.frequency.unit})')
        ax.set_ylabel('Magnitude (dB)')
        ax.legend(loc = 'upper right')

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('FER6 Differential to common CDL - IL')
        s1 = mm_2xthru.s_db[:, 2, 0] - mm_2xthru.s_db[:, 1, 0]
        s2 = mm_2xthru.s_db[:, 3, 1] - mm_2xthru.s_db[:, 0, 1]
        ax.plot(mm_2xthru.frequency.f, s1, color = '0.5', label = 'SCD21 - S21')
        ax.plot(mm_2xthru.frequency.f, s2, color = 'k', label = 'SCD42 - S12')
        self.plot_limit_fer6(mm_2xthru.frequency, ax)
        ax.legend(loc = 'upper right')

        fig.tight_layout()
        return fig

    def plot_td_se_fer(self, s2xthru: Network, sfix_dut_fix: Network,
                            fig: Figure = None) -> Figure:
        """
        Plot fixture electrical requirements (FER) for z values
        """
        if fig is None:
            fig = figure(figsize=(8, 8))

        fig.suptitle('Fixture electrical requirements (FER)')
        f = s2xthru.frequency.f
        s2xthru_dc = IEEEP370.extrapolate_to_dc(s2xthru)
        sfix_dut_fix_dc = IEEEP370.extrapolate_to_dc(sfix_dut_fix)
        n = s2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * s2xthru.frequency.step) # ns
        s21 = s2xthru.s[:, 1, 0]
        t21 = fftshift(irfft(s21, n = n))
        x_k = np.argmax(t21) - n//2
        x_t = x_k * dt

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('FER5 TDR Z variation side 1')
        sfix_dut_fix_dc.plot_z_time_step(0, 0, color = 'k', ax = ax)
        s2xthru_dc.plot_z_time_step(0, 0, color = 'k', linestyle = 'dashed', ax = ax)
        x = ax.lines[-1].get_xdata()[:(x_k + n//2 + 1)]
        y = ax.lines[-1].get_ydata()[:(x_k + n//2 + 1)]
        self.plot_limit_fer5(x, y, ax)
        ax.legend(loc = 'lower right')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               1.1 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ymin = np.min(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               0.9 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ax.set_ylim((ymin - 5, ymax + 5))
        delay = 2 * x_t
        ax.set_xlim((-0.5 * delay, 1.5 * delay))

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('FER5 TDR Z variation side 2')
        sfix_dut_fix_dc.plot_z_time_step(1, 1, color = 'k', ax = ax)
        s2xthru_dc.plot_z_time_step(1, 1, color = 'k', linestyle = 'dashed', ax = ax)
        x = ax.lines[-1].get_xdata()[:(x_k + n//2 + 1)]
        y = ax.lines[-1].get_ydata()[:(x_k + n//2 + 1)]
        self.plot_limit_fer5(x, y, ax)
        ax.legend(loc = 'lower right')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               1.1 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ymin = np.min(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               0.9 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ax.set_ylim((ymin - 5, ymax + 5))
        delay = 2 * x_t
        ax.set_xlim((-0.5 * delay, 1.5 * delay))

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title('FER8 TDT minimum length')
        s2xthru_dc.plot_z_time_impulse(1, 0, color = '0.5', ax = ax)
        s2xthru_dc.plot_z_time_impulse(1, 0, color = 'k', ax = ax)
        y = ax.lines[-1].get_ydata()
        y_lim = [np.min(y), np.max(y)]
        t_lim = [3.0e9 / f[-1], 3.0e9 / f[-1]]
        ax.plot([0, 0], y_lim, color = 'b', linestyle = 'dashed', label = 'Start')
        ax.plot(t_lim, y_lim, color = 'r', linestyle = 'dashed', label = 'Minimum A, B, C')
        ax.legend(loc = 'upper right')
        ax.set_xlim((-1, x_t + 1))

        fig.tight_layout()
        return fig

    def plot_td_mm_fer(self, s2xthru: Network, sfix_dut_fix: Network,
                            fig: Figure = None) -> Figure:
        """
        Plot fixture electrical requirements (FER) for z values
        """
        if fig is None:
            fig = figure(figsize=(8, 8))

        mm_2xthru = s2xthru.copy()
        mm_2xthru.se2gmm(p=2)
        mm_fix_dut_fix = sfix_dut_fix.copy()
        mm_fix_dut_fix.se2gmm(p=2)

        fig.suptitle('Fixture electrical requirements (FER)')
        f = mm_2xthru.frequency.f
        se_2xthru_dc = IEEEP370.extrapolate_to_dc(s2xthru)
        mm_2xthru_dc = IEEEP370.extrapolate_to_dc(mm_2xthru)
        mm_fix_dut_fix_dc = IEEEP370.extrapolate_to_dc(mm_fix_dut_fix)
        n = mm_2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * mm_2xthru.frequency.step) # ns
        s21 = mm_2xthru.s[:, 1, 0]
        t21 = fftshift(irfft(s21, n = n))
        x_k = np.argmax(t21) - n//2
        x_t = x_k * dt

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('FER5 TDR Z variation side 1')
        mm_fix_dut_fix_dc.plot_z_time_step(0, 0, color = 'k', ax = ax)
        mm_2xthru_dc.plot_z_time_step(0, 0, color = 'k', linestyle = 'dashed', ax = ax)
        x = ax.lines[-1].get_xdata()[:(x_k + n//2 + 1)]
        y = ax.lines[-1].get_ydata()[:(x_k + n//2 + 1)]
        self.plot_limit_fer5(x, y, ax)
        ax.legend(loc = 'lower right')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               1.1 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ymin = np.min(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               0.9 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ax.set_ylim((ymin - 5, ymax + 5))
        delay = 2 * x_t
        ax.set_xlim((-0.5 * delay, 1.5 * delay))

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('FER5 TDR Z variation side 2')
        mm_fix_dut_fix_dc.plot_z_time_step(1, 1, color = 'k', ax = ax)
        mm_2xthru_dc.plot_z_time_step(1, 1, color = 'k', linestyle = 'dashed', ax = ax)
        x = ax.lines[-1].get_xdata()[:(x_k + n//2 + 1)]
        y = ax.lines[-1].get_ydata()[:(x_k + n//2 + 1)]
        self.plot_limit_fer5(x, y, ax)
        ax.legend(loc = 'lower right')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               1.1 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ymin = np.min(np.array([ax.lines[0].get_ydata()[(n // 2):(x_k + n // 2)],
                               0.9 * ax.lines[1].get_ydata()[(n // 2):(x_k + n // 2)]]))
        ax.set_ylim((ymin - 5, ymax + 5))
        delay = 2 * x_t
        ax.set_xlim((-0.5 * delay, 1.5 * delay))

        ax = fig.add_subplot(2, 2, 3)
        ax.set_title('FER7 TDT skew')
        se_2xthru_dc.plot_z_time_impulse(2, 0, color = '0.5', ax = ax)
        se_2xthru_dc.plot_z_time_impulse(3, 1, color = 'k', ax = ax)
        ax.legend(loc = 'upper right')
        ax.set_xlim((-1, x_t + 1))

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('FER8 TDT minimum length')
        mm_2xthru_dc.plot_z_time_impulse(1, 0, color = '0.5', ax = ax)
        mm_2xthru_dc.plot_z_time_impulse(1, 0, color = 'k', ax = ax)
        y = ax.lines[-1].get_ydata()
        y_lim = [np.min(y), np.max(y)]
        t_lim = [3.0e9 / f[-1], 3.0e9 / f[-1]]
        ax.plot([0, 0], y_lim, color = 'b', linestyle = 'dashed', label = 'Start')
        ax.plot(t_lim, y_lim, color = 'r', linestyle = 'dashed', label = 'Minimum A, B, C')
        ax.legend(loc = 'upper right')
        ax.set_xlim((-1, x_t + 1))

        fig.tight_layout()
        return fig

class IEEEP370_FD_QM:
    def __init__(self, verbose: bool = False) -> None:
        """
        IEEE 370 initial quality checking of raw data at the given frequency
        samples.

        Initial passivity (PQMi), reciprocity (RQMi), and causality (CQMi)
        checks are performed on the original S-parameters data in the frequency
        domain.

        Based on [IEEE370]_.

        Parameters
        ----------
        verbose      : :boolean
                       Plot internal causality, passivity, and reciprocity
                       figures (default False)

        References
        ----------
        .. [IEEE370] IEEE Standard for Electrical Characterization of Printed
        Circuit Board and Related Interconnects at Frequencies up to 50 GHz",
        IEEE 370-2020.
        """
        self.verbose = verbose

    def check_causality(self, ntwk: Network) -> float:
        """
        Initial Causality Quality Metric (CQMi): verify that the complex
        S-parameters rotate clockwise in the complex plane.

        This is done by computing the normalized vector product on pairs of
        consecutive vectors between two frequency points.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Network to be checked

        Returns
        -------
        PQM : :class:`~skrf.network.Network` object
              Causality quality metric in percents
        """
        Nf = ntwk.frequency.npoints
        CQM = zeros((ntwk.nports, ntwk.nports))
        for i in range(ntwk.nports):
            for j in range(ntwk.nports):
                if len(np.unique(ntwk.s[:, i, j])) == 1:
                    CQM[i, j] = 100.
                else:
                    TotalR = 0
                    PositiveR = 0
                    for k in range(Nf - 2):
                        Vn = ntwk.s[k + 1, i, j] - ntwk.s[k, i, j]
                        Vn1 = ntwk.s[k + 2, i, j] - ntwk.s[k + 1, i, j]
                        R = real(Vn1) * imag(Vn) - imag(Vn1) * real(Vn)
                        if R > 0:
                            PositiveR = PositiveR + R
                        TotalR = TotalR + np.abs(R)
                    CQM[i, j] = np.nanmax((PositiveR / TotalR, 0)) * 100.

        return np.min(CQM)

    def check_passivity(self, ntwk: Network) -> float:
         """
         Initial Passivity Quality Metric (PQMi): verify that the 2-Norm of
         S-parameters is smaller or equal to 1 at each frequency.

         This is equivalent to checking the eigenvalues of the unity matrix
         subtracted by complex conjugate transposed S time S is greater or
         equal to zero.

         Parameters
         ----------
         ntwk: :class:`~skrf.network.Network` object
               Network to be checked

         Returns
         -------
         PQM : :class:`~skrf.network.Network` object
               Passivity quality metric in percents
         """
         if ntwk.nports == 1:
             raise (ValueError('Doesn\'t exist for one-ports'))

         Nf = ntwk.frequency.npoints
         A  = 1.00001
         B  = 0.1
         self.PM = zeros(Nf)
         PW = zeros(Nf)
         for i in range(Nf):
             # numpy linalg norm is frobenius, use 2-norm like Matlab instead
             self.PM[i] = norm(ntwk.s[i, :, :], 2)
             if self.PM[i] > A:
                 PW[i] = (self.PM[i] - A) / B

         return np.max([Nf - np.sum(PW), 0]) / Nf * 100.


    def check_reciprocity(self, ntwk: Network) -> float:
        """
        Initial Reciprocity Quality Metric (RQMi):
        integrates the absolute difference between Sij and Sji at
        each frequency point. Ideally, Sij should be equal to Sji.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Network to be checked

        Returns
        -------
        PQM : :class:`~skrf.network.Network` object
              Reciprocity quality metric in percents
        """
        if ntwk.nports == 1:
            raise (ValueError('Doesn\'t exist for one-ports'))

        Nf = ntwk.frequency.npoints
        B = 0.1
        C = 1e-6
        self.RM = zeros(Nf)
        RW = zeros(Nf)
        for i in range(Nf):
            self.RM[i] = 0
            for k in range(ntwk.nports):
                for m in range(ntwk.nports):
                    self.RM[i] = self.RM[i] + np.abs(ntwk.s[i, k, m] - ntwk.s[i, m, k])
            self.RM[i] = self.RM[i] / (ntwk.nports * (ntwk.nports - 1))
            if self.RM[i] > C:
                RW[i] = (self.RM[i] - C) / B

        return np.max([Nf - np.sum(RW), 0]) / Nf * 100.

    def check_se_quality(self, ntwk: Network, verbose: bool = False) -> dict:
        """
        Single-ended passivity, reciprocity, and causality checks are
        performed on the original S-parameters data in the frequency domain.

        Parameters
        ----------
        ntwk   : :class:`~skrf.network.Network` object
                 Network to be checked
        verbose: :boolean
                 Plot internal causality, passivity, and reciprocity
                 figures. When True, override class verbose parameter.
                 (default False).

        Returns
        -------
        QM : :class:`dict` object
              Dictionary with quality metrics
        """
        verbose = self.verbose or verbose
        QM = {'causality': {'value': self.check_causality(ntwk), 'evaluation': ''},
              'passivity': {'value': self.check_passivity(ntwk), 'evaluation': ''},
              'reciprocity': {'value': self.check_reciprocity(ntwk), 'evaluation': ''},
              }

        # evaluation
        if QM['causality']['value'] <= 20.:
            QM['causality']['evaluation'] = 'poor'
        elif QM['causality']['value'] <= 50.:
            QM['causality']['evaluation'] = 'inconclusive'
        elif QM['causality']['value'] <= 80:
            QM['causality']['evaluation'] = 'acceptable'
        else:
            QM['causality']['evaluation'] = 'good'

        if QM['passivity']['value'] <= 80.:
            QM['passivity']['evaluation'] = 'poor'
        elif QM['passivity']['value'] <= 99.:
            QM['passivity']['evaluation'] = 'inconclusive'
        elif QM['passivity']['value'] <= 99.9:
            QM['passivity']['evaluation'] = 'acceptable'
        else:
            QM['passivity']['evaluation'] = 'good'

        if QM['reciprocity']['value'] <= 80.:
            QM['reciprocity']['evaluation'] = 'poor'
        elif QM['reciprocity']['value'] <= 99.:
            QM['reciprocity']['evaluation'] = 'inconclusive'
        elif QM['reciprocity']['value'] <= 99.9:
            QM['reciprocity']['evaluation'] = 'acceptable'
        else:
            QM['reciprocity']['evaluation'] = 'good'

        # verbose
        if verbose:
            name = ntwk.name if ntwk.name else 'Network'
            fig = figure(figsize = (12, 4.4))
            fig.suptitle('Initial checking in the frequency domain')
            ax = fig.add_subplot(1, 3, 1, projection = 'polar')
            ax.set_title('Causality')
            ntwk.plot_s_polar(ax = ax)
            ax.legend(loc = 'upper right')
            ax = fig.add_subplot(1, 3, 2)
            ax.set_title('Passivity')
            ax.plot(ntwk.frequency.f_scaled, self.PM, color = 'k', label = name)
            ax.plot([ntwk.frequency.f_scaled[0], ntwk.frequency.f_scaled[-1]],
                    [1., 1.], color = 'r', linestyle = 'dashed', label = 'Maximum')
            ax.set_xlabel(f'Frequency ({ntwk.frequency.unit})')
            ax.set_ylabel('2-Norm(S)')
            ax.legend(loc = 'upper right')
            ax = fig.add_subplot(1, 3, 3)
            ax.set_title('Reciprocity')
            ax.plot(ntwk.frequency.f_scaled, self.RM, color = 'k', label = name)
            ax.set_xlabel(f'Frequency ({ntwk.frequency.unit})')
            ax.set_ylabel('Sum of S-pairs differences')
            ax.legend(loc = 'upper right')
            fig.tight_layout()

        return QM

    def check_mm_quality(self, ntwk: Network, verbose: bool = False) -> dict:
        """
        Mixed-mode passivity, reciprocity, and causality checks are
        performed on the original S-parameters data in the frequency domain.

        The input networks should be 4-port single-ended and will be
        transformed to mixed-mode representation. Only the differential and
        the common modes are tested.

        Parameters
        ----------
        ntwk:    :class:`~skrf.network.Network` object
                 Network to be checked
        verbose: :bool
                 Plot internal causality, passivity, and reciprocity
                 figures. When True, override class verbose parameter.
                 (default False).

        Returns
        -------
        QM : :class:`dict` object
              Dictionary with quality metrics
        """
        mm = ntwk.copy()
        mm.se2gmm(p = 2)
        QM = {'dd': self.check_se_quality(mm.subnetwork([0, 1]), verbose),
              'cc': self.check_se_quality(mm.subnetwork([2, 3]), verbose)}

        return QM

    def print_qm(self, QM: dict) -> dict:
        """
        Print the quality metrics dictionary.

        Parameters
        ----------
        QM: :class:`dict` object
            Dictionary with quality metrics to print
        """
        # Ignoring T201 for QM printout as it's not a user-facing print statement
        if 'dd' in QM:
            print('Differential mode') # noqa: T201
            for k in QM['dd'].keys():
                print(f"{k} is {QM['dd'][k]['evaluation']} ({QM['dd'][k]['value']:.2f}%)") # noqa: T201
            print('Common mode') # noqa: T201
            for k in QM['cc'].keys():
                print(f"{k} is {QM['cc'][k]['evaluation']} ({QM['cc'][k]['value']:.2f}%)") # noqa: T201
        else:
            for k in QM.keys():
                print(f"{k} is {QM[k]['evaluation']} ({QM[k]['value']:.2f}%)") # noqa: T201

class IEEEP370_TD_QM:
    def __init__(self, data_rate: float, sample_per_UI: int,
                 rise_time_per: float, pulse_shape: int = 1,
                 extrapolation: int = 2, verbose: bool = False) -> None:
        """
        IEEEP370_TD_QM Application-based quality checking of in the time
        domain using causality (CQMa), passivity (PQMa), and reciprocity (RQMa)
        metrics.

        If necessary, the original S-parameters are extrapolated to a frequency
        of three times the desired data rate. Causal, passive, and reciprocal
        models are reconstructed and stimulated by a pulse. The difference
        between the original extrapolated signal and the ideal responses
        in the time domain are integrated to give metrics in millivolts.

        Based on [IEEE370]_.

        Parameters
        -----------
        data_rate    : :float
                       Data rate (bps)
        sample_per_UI: :number
                       Number of points of unit interval
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width
        pulse_shape  : :number
                       1 is Gaussian; 2 is rectangular with Butterworth filter;
                       3 is rectangular with Gaussian filter
        extrapolation: :number
                       1 is constant extrapolation; 2 is zero padding
        verbose      : :boolean
                       Plot extrapolated frequency data, generated pulse and
                       the time domain comparison between the original and the
                       causality enforced responses

        References
        ----------
        .. [IEEE370] IEEE Standard for Electrical Characterization of Printed
        Circuit Board and Related Interconnects at Frequencies up to 50 GHz",
        IEEE 370-2020.
        """
        self.data_rate = data_rate
        self.sample_per_UI = sample_per_UI
        self.rise_time_per = rise_time_per
        self.pulse_shape = pulse_shape
        self.extrapolation = extrapolation
        self.verbose = verbose

    def add_conj(self, s_ij: ndarray):
        """
        Add complex conjugates for ifft.

        Todo: Consider using irfft instead.
        """
        N = len(s_ij)
        s_ij_conj = zeros(2 * N - 1, dtype = complex)
        s_ij_conj[:N] = s_ij
        for k in range(N - 1):
            s_ij_conj[k + N] = np.conj(s_ij_conj[N - k - 1])

        return s_ij_conj

    def align_signals(self, x: ndarray, y: ndarray) -> ndarray:
        """
        Compute the index shift between two identical shifted signals in the
        time domain.
        """
        y = y.T
        x = x.T
        n = len(x)
        m = np.round(n * 0.1).astype(int)
        mm = np.round(n * 0.01).astype(int)
        xx = np.append(x[0:m], x[n - mm:n])
        yy = np.append(y[0:m], y[n - mm:n])
        x = xx
        y = yy
        yy = y[0:m]
        Ix = np.argmax(x)
        Iy = np.argmax(y)
        index = Ix - Iy
        yy = np.roll(y, index)
        n = np.min([1000, m]).astype(int)
        error = len(x)
        error_ind = 0
        for k in range(-n + index, n + index):
            yy = np.roll(y, k)
            # numpy linalg norm is frobenius, use 2-norm like Matlab instead
            cur_error = np.linalg.norm(yy - x, 2)
            if error > cur_error:
                error_ind = k
                error = cur_error
        y = np.roll(y, error_ind)

        return error_ind

    def create_causal(self, ntwk: Network, data_rate: float,
                      rise_time_per: float) -> (Network, ndarray):
        """
        Create a causality-enforced network.

        Parameters
        ----------
        ntwk         : :class:`~skrf.network.Network` object
                       Input network
        data_rate    : :float
                       Data rate (bps)
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width

        Returns
        -------
        causal: :class:`~skrf.network.Network` object
                Causality-enforced network
        delay : :class:`~skrf.network.Network` object
                Alignment delay with original network for comparison sake
        """
        causal = ntwk.copy()
        nports = causal.nports
        N = causal.frequency.npoints
        f = causal.frequency.f
        delay_matrix = zeros((nports, nports), dtype = int)
        for i in range(nports):
            for j in range(nports):
                for k in range(N):
                    if np.abs(causal.s[k, i, j]) == 0:
                        causal.s[k, i, j] = 0.00001
                causal_ij, f, delay_ij = self.get_causal_model(f, causal.s[:, i, j],
                                                               data_rate,
                                                               rise_time_per)
                causal.s[:, i, j] = causal_ij
                delay_matrix[i, j] = delay_ij

        return (causal, delay_matrix)

    def create_passive(self, ntwk: Network) -> Network:
        """
        Creat a passivity-enforced network.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Input network

        Returns
        -------
        reciprocal : :class:`~skrf.network.Network` object
                     Passivity-enforced network
        """
        passive = ntwk.copy()
        for i in range(ntwk.frequency.npoints):
            U, D, Vh = np.linalg.svd(ntwk.s[i, :, :])
            for k in range(ntwk.nports):
                if D[k] > 1.:
                    D[k] = 1.
            passive.s[i, :, :] = U @ np.diag(D) @ Vh

        return passive

    def create_reciprocal(self, ntwk: Network) -> Network:
        """
        Creat a reciprocal network.

        The resulting network is the reciprocal of the input networks. The
        reciprocity is not enforced, but the time domain response will still
        gives metrics that is high if reciprocity is low.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Input network

        Returns
        -------
        reciprocal : :class:`~skrf.network.Network` object
                     Reciprocal network
        """
        reciprocal = ntwk.copy()
        for i in range(ntwk.nports):
            for j in range(ntwk.nports):
                reciprocal.s[:, i, j] = ntwk.s[:, j, i]

        return reciprocal

    def extrapolate_to_dc(self, ntwk: Network) -> Network:
        """
        Extrapolate to DC and interpolate to the harmonic frequency sweep.

        Passivity is enforced on the DC extrapolated points. The missing part
        from dc to the first frequency sample should not exceed a phase change
        of π/2.

        Parameters
        ----------
        ntwk: :class:`~skrf.network.Network` object
              Input network

        Returns
        -------
        extrapolated : :class:`~skrf.network.Network` object
                       Extrapolated network
        """
        f = ntwk.frequency.f
        df = f[1] - f[0]
        nports = ntwk.nports
        f_0 = f[0]
        # numpy linalg norm is frobenius, use 2-norm like Matlab instead
        norm_0 = np.linalg.norm(ntwk.s[0, :, :], 2)
        if f[0] == 0:
            f_extra = f
        else:
            f_new = df * np.arange(0, np.ceil(f[0] / df))
            f_extra = np.append(f_new, f)
        N_interp = np.floor(f_extra[-1]/df)
        f_interp = df * np.arange(0, N_interp + 1)
        s = zeros((len(f_extra), nports, nports), dtype = complex)
        s_interp = zeros((len(f_interp), nports, nports), dtype = complex)
        for i in range(nports):
            for j in range(nports):
                # dc extrapolation
                if f[0] == 0:
                    s[:, i, j] = ntwk.s[:, i, j]
                    s[0] = np.real(s[0, i, j])
                else:
                    s[:, i, j] = self.extrapolate_to_dc_ij(f, f_new,
                                              ntwk.s[:, i, j])
                    # interpolate to the harmonic sweep
                    s_interp[:, i, j] = self.interpolate_ij(f_extra, f_interp,
                                                   s[:, i, j])
        # enforce passivity of extrapolated points
        i = 0
        D_max = np.max(np.array([1., norm_0]))
        while f_interp[i] < f_0:
            U, D, Vh = np.linalg.svd(s_interp[i, :, :])
            for k in range(nports):
                if D[k] > D_max:
                    D[k] = D_max
            s_interp[i, :, :] = U @ np.diag(D) @ Vh
            i += 1

        return Network(frequency = f_interp, s = s_interp, name = ntwk.name,
                       z0 = ntwk.z0[0])

    def extrapolate_to_dc_ij(self, f: ndarray, f_new: ndarray, s_ij: ndarray):
        """
        Extrapolate single S-component to DC.

        The missing part from dc to the first frequency sample should not
        exceed a phase change of π/2.
        """
        # calculate delay
        ph = -np.unwrap(np.angle(s_ij))
        delay = self.get_delay(f, ph)
        # extract delay to smooth original function
        s_ij = s_ij * np.exp(1j * 2 * np.pi * f * delay)
        # extract real and imaginary parts from the original function
        re = np.real(s_ij)
        im = np.imag(s_ij)
        # create a*x^2+b parabola using (f(1),re(1)) and (f(2),re(2)) points
        a = (re[1] - re[0]) / (f[1]**2 - f[0]**2)
        b = re[0] - a * f[0]**2
        # extend real part to DC
        re_new = a * f_new**2 + b
        re = np.append(re_new, re)
        # create a*x^3+b*x cubic parabola using (f(1),im(1)) and (f(2),im(2)) points
        a = (im[1]/f[1] - im[0]/f[0])/(f[1]**2 - f[0]**2)
        b = im[0]/f[0] - a * f[0]**2
        # extend imaginary part to DC
        im_new = a * f_new**3 + b * f_new
        im = np.append(im_new, im)
        f_extra = np.append(f_new, f)
        # create complex function from real and imaginary parts
        s_ij_extra = re + 1j * im
        # return delay
        s_ij_extra = s_ij_extra * \
            np.exp(-1j * 2 * np.pi * f_extra * delay)

        return s_ij_extra

    def extrapolate_to_fmax(self, ntwk: Network, data_rate: float,
                            sample_per_UI: int, extrapolation: int)-> Network:
        """
        Extrapolate network max frequency if required by parameters.

        This is usually three times the data rate.

        Parameters
        ----------
        ntwk         : :class:`~skrf.network.Network` object
                       Input network
        data_rate    : :float
                       Data rate (bps)
        sample_per_UI: :number
                       Number of points of generated pulse signal
        rise_time_per: :float
                       Rise time divided by high time ratio
        extrapolation: :number
                       1 is constant extrapolation; 2 is zero padding

        Returns
        -------
        extrapolated : :class:`~skrf.network.Network` object
                       Extrapolated network
        """
        f_max = 0.5 * data_rate * sample_per_UI
        df = ntwk.frequency.f[1] - ntwk.frequency.f[0]
        f_new = ntwk.frequency.f
        while(f_new[-1] < f_max):
            f_new = np.append(f_new, f_new[-1] + df)
        N1 = ntwk.frequency.npoints
        N = len(f_new)
        s_new = zeros((N, ntwk.nports, ntwk.nports), dtype = complex)
        for i in range(ntwk.nports):
            for j in range(ntwk.nports):
                s_new[:N1, i, j] = ntwk.s[:, i, j]
                ph = np.unwrap(np.angle(s_new[:N1, i, j]))
                dph = (ph[-1] - ph[0]) / (N1 - 1)
                for k in range(N1, N):
                    if extrapolation == 1:
                        s_new[k, i, j] = s_new[k - 1, i, j] * np.exp(1j * dph)
                    else:
                        s_new[k, i, j] = 0

        return Network(frequency = f_new, s = s_new, name = ntwk.name,
                       z0 = ntwk.z0[0])


    def get_causal_model(self, f: ndarray, s_ij: ndarray, data_rate,
                         rise_time_per) -> ndarray:
        """
        Get causality-enforced model for a single S-parameter component Sij.
        """
        df = f[1] - f[0]
        dt = 1. / (2 * f[-1] + df)
        # DC extrapolation
        # Already done.
        # Interpolate data
        # Already done.
        # extend to negative frequencies
        N = len(s_ij)
        s_ij[0] = np.real(s_ij[0])
        s_ij_conj = self.add_conj(s_ij)
        # Extract magnitude
        s_ij_magn_conj = np.real(np.log(np.abs(s_ij_conj)))
        # Convert magnitude into time domain
        s_ij_magn_time = np.fft.ifft(s_ij_magn_conj)
        # Multiply by sign(t)
        for i in range(N, 2 * N - 1):
            s_ij_magn_time[i] = (-1.) * s_ij_magn_time[i]
        s_ij_magn_time = 1j * s_ij_magn_time
        # Calculate Phase
        s_ij_phase_enforced = np.real(fft(s_ij_magn_time))
        # Calculate Delay
        delay = self.get_delay_time(f, s_ij, s_ij_phase_enforced[0:N],
                                    data_rate, rise_time_per)
        delay = np.round(delay / dt) * dt
        causal_ij = zeros(N, dtype = complex)
        for i in range(N):
            w = 2 * np.pi * f[i]
            causal_ij[i] = np.exp(s_ij_magn_conj[i]) * \
                np.exp(-1j * s_ij_phase_enforced[i]) * np.exp(-1j * delay * w)
        delay = np.round(delay / dt).astype(int)
        return (causal_ij, f, delay)


    def get_delay(self, freq: ndarray, phase: ndarray) -> float:
        """
        Get the front delay from phase and frequency vectors.

        Parameters
        ----------
        freq: :ndarray
              Frequency (Hz)
        phase: :ndarray
               Phase (rad)

        Returns
        -------
        delay: :float
               Delay

        """
        N = len(freq)
        delay = 1.
        for i in range(N):
            if freq[i] > 0:
                delay_i = phase[i] / freq[i] / 2. / np.pi
                if delay > delay_i:
                    delay = delay_i
        return delay

    def get_delay_time(self, freq: ndarray, s_ij: ndarray, phase_causal: ndarray,
                                data_rate: float, rise_time_per: float) -> float:
        """
        Get delay between original and causality enforced data in number of
        time samples.

        Parameters
        ----------
        freq         : :ndarray
                       Frequency (Hz)
        s_ij         : :ndarray
                       Original single S-component.
        phase_causal : :ndarray
                       Causality enforced phase (rad)
        data_rate    : :float
                       Data rate (bps)
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width

        Returns
        -------
        delay: :float
               Delay
        """
        N = len(freq)
        df = freq[1] - freq[0]
        dt = 1. / (2 * freq[-1] + df)
        # Gaussian filter
        f_cut = 3. * data_rate / 2.
        sigma = 1. / 2. / np.pi / f_cut
        gaussian = np.exp(-2 * np.pi * np.pi * freq * freq * sigma * sigma)
        original = s_ij * gaussian
        causal = np.abs(original) * np.exp(-1j * phase_causal)
        original_conj = self.add_conj(original)
        causal_conj = self.add_conj(causal)
        pulse = self.get_pulse_rect(dt, data_rate, 2 * N - 1, rise_time_per)
        pulse_original = original_conj * pulse
        pulse_causal = causal_conj * pulse
        v_origin = np.fft.ifft(pulse_original) / 2.
        v_causal = np.fft.ifft(pulse_causal) / 2.
        shift_ind = -1 * self.align_signals(v_causal, v_origin)
        return shift_ind * dt

    def get_pulse_gaussian(self, dt: float, data_rate: float, N: int,
                         rise_time_per: float, verbose = False) -> ndarray:
        """
        Get the FFT of a gaussian pulse. The pulse is shifted in time according
        to parameters.

        Parameters
        ----------
        dt           : :float
                       Sample time (s)
        data_rate    : :float
                       Data rate (bps)
        N            : :number
                       Number of points of generated pulse signal
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width
        verbose      : :boolean
                       Plot reference and generated pulses in the time
                       domain

        Returns
        -------
        fft : :ndarray
              FFT of the pulse signal
        """
        n_samples = (N - 1) // 2
        self.t_pulse = np.arange(-n_samples, n_samples + 1) * dt
        N = len(self.t_pulse)
        sigma = rise_time_per / (data_rate * \
                                 (np.sqrt(-np.log(0.2))-np.sqrt(-np.log(0.8))))
        self.v_ref = np.exp(-self.t_pulse**2 / sigma**2)
        k_middle = n_samples
        k_start  = np.round(1.5 / data_rate / dt) - 1
        self.v_pulse = zeros(self.t_pulse.shape)
        for i in range(N):
            self.v_pulse[i] = self.v_ref[np.mod(i + k_middle - k_start, N).astype(int)]
        if verbose:
            fig, ax = subplots(1, 1)
            ax.plot(self.t_pulse, self.v_ref, color = 'r', label = 'Reference')
            ax.plot(self.t_pulse, self.v_pulse, linestyle = 'dashed', label = 'Generated')
            ax.legend(loc = 'upper right')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Gaussian Pulse')

        return fft(self.v_pulse)

    def get_pulse_rect(self, dt: float, data_rate: float, N: int,
                       rise_time_per: float, verbose = False)-> ndarray:
        """
        Get the FFT of a rectangular pulse with defined rise time.

        Rise time and fall time are equals.

        Parameters
        ----------
        dt           : :float
                       Sample time (s)
        data_rate    : :float
                       Data rate (bps)
        N            : :number
                       Number of points of generated pulse signal
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width
        verbose      : :boolean
                       Plot reference and interpolated pulses in the time
                       domain

        Returns
        -------
        fft : :ndarray
              FFT of the pulse signal
        """
        self.t_pulse = np.arange(0, N) * dt
        k_high = np.round(1. / data_rate / dt)
        k_rise = np.round(k_high * rise_time_per)
        k_offset = np.array([0, k_rise, k_rise, k_rise, k_rise, 0])
        k_ref = np.array([0, 0, k_rise, k_high, k_high + k_rise, N - 1])
        self.t_ref = dt * (k_offset + k_ref)
        self.v_ref = np.array([0, 0, 1, 1, 0, 0])

        interp = scipy.interpolate.interp1d(self.t_ref, self.v_ref)
        self.v_pulse = interp(self.t_pulse)

        if verbose:
            fig, ax = subplots(1, 1)
            ax.plot(self.t_ref, self.v_ref, color = 'r', marker = 'o',
                    label = 'Reference')
            ax.plot(self.t_pulse, self.v_pulse, linestyle = 'dashed',
                    label = 'Interpolated')
            ax.legend(loc = 'upper right')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Rectangular Pulse')

        return fft(self.v_pulse)

    def get_time_domain(self, ntwk: Network, data_rate: float,
                        rise_time_per: float,
                        pulse_shape: int) -> (ndarray, ndarray):
        """
        Get the impulse responses of the S-parameters.

        The pulse is defined as per application parameters.

        Parameters
        ----------
        ntwk         : :class:`~skrf.network.Network` object
                       Input network
        data_rate    : :float
                       Data rate (bps)
        rise_time_per: :float
                       Rise time from 20% to 80% divided by width
        pulse_shape  : :number
                       1 is Gaussian; 2 is rectangular with Butterworth filter;
                       3 is rectangular with Gaussian filter

        Returns
        -------
        v : :ndarray
            Impulse response amplitude vector
        t : :ndarray
            Impulse response time vector
        """
        N = ntwk.frequency.npoints
        freq = ntwk.frequency.f
        df = freq[1] - freq[0]
        dt = 1 / (2 * freq[-1] + df)
        nports = ntwk.nports
        t = dt * np.arange(0, 2 * N - 1)
        v = zeros((2 * N - 1, nports, nports))
        # Gaussian filter
        f_cut = 3. * data_rate / 2.
        sigma = 1. / 2. / np.pi / f_cut
        rise_time = 1. / data_rate * 1000 * rise_time_per
        f0 = 320 / rise_time
        if pulse_shape == 1:
            self.filter = np.ones(N, dtype = complex)
            self.pulse = self.get_pulse_gaussian(dt, data_rate, 2 * N - 1,
                                            rise_time_per)
        elif pulse_shape == 2:
            self.filter = 1. / (1 + 1j * freq / f0)
            self.pulse = self.get_pulse_rect(dt, data_rate, 2 * N - 1,
                                        1.4 * rise_time_per)
        else:
            self.filter = np.exp(-2 * np.pi * np.pi * freq * freq * sigma * sigma)
            self.pulse = self.get_pulse_rect(dt, data_rate, 2 * N - 1,
                                        1.4 * rise_time_per)
        for i in range(nports):
            for j in range(nports):
                s_ij = ntwk.s[:, i, j] * self.filter
                s_ij[0] = np.real(s_ij[0])
                s_ij_conj = self.add_conj(s_ij)
                pulse_response_freq = self.pulse * s_ij_conj
                v[:, i, j] = np.real(np.fft.ifft(pulse_response_freq))

        return (v, t)

    def get_td_difference_mv(self, v1: ndarray, v2: ndarray, t: ndarray,
                                      nports: int,
                                      data_rate: float) -> (ndarray, ndarray):
        """
        Integrate the difference between two signals on units intervals in the
        time domain. The result has a physical estimation in millivolts of
        the worst-case bit sequence based on peak distortion analysis.
        """
        N = len(t)
        dt = t[1] - t[0]
        UI = 1. / data_rate / dt
        max_bits = 31
        time_domain_difference_mv = zeros((nports, nports))
        N_UI = np.round(UI).astype(int)
        delta = zeros(N_UI)
        for i in range(nports):
            for j in range(nports):
                max_index = np.argmax(v1[:, i, j])
                last_index = max_index + max_bits * UI
                lower_index = max_index - max_bits * UI
                for k in range(N_UI):
                    delta[k] = 0
                    for m in range(np.floor(N / UI).astype(int) - 1):
                        ind = k + np.floor(m * UI).astype(int)
                        if lower_index >= 0:
                            condition = (ind < last_index) and (ind > lower_index)
                        else:
                            condition = (ind < last_index) or (ind > N - lower_index - 1)
                        if condition:
                            delta[k] = delta[k] + np.abs(v2[ind, i, j] - v1[ind, i, j])
                time_domain_difference_mv[i, j] = np.max(delta)

        return time_domain_difference_mv

    def get_td_causality_difference_mv(self, v1: ndarray, v2: ndarray, t: ndarray,
                                      nports: int, data_rate: float,
                                      delay_matrix: ndarray) -> (ndarray, ndarray):
        """
        Integrate the difference between two signals on units intervals in the
        time domain. The result has a physical estimation in millivolts of
        the worst-case bit sequence based on peak distortion analysis.

        Also accounts for the causality minimum delay.
        """
        N = len(t)
        dt = t[1] - t[0]
        UI = 1. / data_rate / dt
        max_bits = 31
        time_domain_difference_mv = zeros((nports, nports))
        N_UI = np.round(UI).astype(int)
        delta = zeros(N_UI)
        for i in range(nports):
            for j in range(nports):
                if i == j:
                    delay_num = 0
                else:
                    delay_num = delay_matrix[i, j]
                for k in range(N_UI):
                    delta[k] = 0
                    for m in range(max_bits):
                        ind = delay_num - k - np.floor(m * UI).astype(int) - 2
                        if ind < 0:
                            ind = N + ind
                        delta[k] = delta[k] + np.abs(v2[ind, i, j] - v1[ind, i, j])
                time_domain_difference_mv[i, j] = np.max(delta)

        return time_domain_difference_mv

    def interpolate_ij(self, f: ndarray, f_new: ndarray, s_ij: ndarray):
        """
        Interpolate single S-parameter component Sij.
        """
        # calculate delay
        delay = np.max([0, self.get_delay(f, -np.unwrap(np.angle(s_ij)))])
        # extract delay to smooth original function
        s_ij = s_ij * np.exp(1j * 2 * np.pi * f * delay)
        # interpolate
        interp = scipy.interpolate.interp1d(f, s_ij)
        s_ij_interp = interp(f_new)
        # return delay
        s_ij_interp = s_ij_interp * \
            np.exp(-1j * 2 * np.pi * f_new * delay)

        return s_ij_interp

    def check_se_quality(self, ntwk: Network, verbose: bool = False) -> dict:
        """
        Single-ended application-based quality checking in the time domain
        using causality (CQMa), passivity (PQMa), and reciprocity (RQMa)
        metrics.

        If necessary, the original S-parameters are extrapolated to a frequency
        of three times the desired data rate. Causal, passive, and reciprocal
        models are reconstructed and stimulated by a pulse. The difference
        between the original extrapolated signal and the ideal responses
        in the time domain are integrated to give metrics in millivolts.

        Parameters
        ----------
        ntwk   : :class:`~skrf.network.Network` object
                 Network to be checked
        verbose: :bool
                 Plot internal causality, passivity, and reciprocity
                 figures. When True, override class verbose parameter.
                 (default False).

        Returns
        -------
        QM : :class:`dict` object
              Dictionary with quality metrics
        """
        verbose = self.verbose or verbose
        if (1.5 * self.data_rate) > ntwk.frequency.f[-1]:
            warnings.warn('Maximum frequency is less then recommended frequency.',
                          RuntimeWarning, stacklevel=2)

        # extrapolate max freq
        self.ntwk_interpolated = self.extrapolate_to_fmax(ntwk, self.data_rate,
                                                     self.sample_per_UI,
                                                     self.extrapolation)

        # extrapolate dc and interpolate with uniform step
        self.ntwk_interpolated = self.extrapolate_to_dc(self.ntwk_interpolated)

        if verbose:
            fig, axs = subplots(2, 3, figsize = (12, 7))
            fig.suptitle('Application-based checking in the time domain')
            ax = axs[0, 0]
            ax.set_title('Extrapolation')
            self.ntwk_interpolated.frequency.unit = ntwk.frequency.unit
            # avoid log(0) issues with zero padding
            ntwk.plot_s_db(1, 0, color = 'r', ax = ax, label = 'Original, S21')
            if self.extrapolation == 2:
                nz_k = np.nonzero(self.ntwk_interpolated.s[:, 1, 0])[0]
                self.ntwk_interpolated[:nz_k[-1]].plot_s_db(1, 0, color = 'b',
                                            linestyle = 'dashed', ax = ax,
                                            label = 'Extrapolated, S21')
            else:
                self.ntwk_interpolated.plot_s_db(1, 0, color = 'b',
                                                 linestyle = 'dashed', ax = ax,
                                                 label = 'Extrapolated, S21')
            secax = ax.twinx()
            ntwk.plot_s_deg(1, 0, color = 'm', ax = secax, label = 'Original, S21')
            self.ntwk_interpolated.plot_s_deg(1, 0, color = 'c',
                                              linestyle = 'dashed', ax = secax,
                                              label = 'Extrapolated, S21')
            ax.legend(loc = 'upper left')
            secax.legend(loc = 'lower right')
            fig.tight_layout()

        # get Causal Matrix
        self.causal, self.delay_matrix = self.create_causal(self.ntwk_interpolated,
                                                  self.data_rate, self.rise_time_per)
        # get Passive Matrix
        self.passive = self.create_passive(self.ntwk_interpolated)
        # get Reciprocal Matrix
        self.reciprocal = self.create_reciprocal(self.ntwk_interpolated)

        # get Time Domain Matrices
        self.v_causal, self.t_causal = self.get_time_domain(self.causal,
                                                  self.data_rate,
                                                  self.rise_time_per,
                                                  self.pulse_shape)
        self.v_passive, self.t_passive = self.get_time_domain(self.passive,
                                                  self.data_rate,
                                                  self.rise_time_per,
                                                  self.pulse_shape)
        self.v_reciprocal, self.t_reciprocal = self.get_time_domain(self.reciprocal,
                                                  self.data_rate,
                                                  self.rise_time_per,
                                                  self.pulse_shape)
        self.v_origin, self.t_origin = self.get_time_domain(self.ntwk_interpolated,
                                                  self.data_rate,
                                                  self.rise_time_per,
                                                  self.pulse_shape)

        # get Time Domain Difference
        self.causality_difference_mv = self.get_td_causality_difference_mv(
            self.v_causal, self.v_origin, self.t_origin,
            2, self.data_rate, self.delay_matrix)
        self.passivity_difference_mv = self.get_td_difference_mv(
            self.v_passive, self.v_origin, self.t_origin, 2, self.data_rate)
        self.reciprocity_difference_mv = self.get_td_difference_mv(
            self.v_reciprocal, self.v_origin, self.t_origin, 2, self.data_rate)

        # numpy linalg norm is frobenius, use 2-norm like Matlab instead
        causality_metric = np.round(
            1000 * np.linalg.norm(self.causality_difference_mv, 2), 1)
        passivity_metric = np.round(
            1000 * np.linalg.norm(self.passivity_difference_mv, 2), 1)
        reciprocity_metric = np.round(
            1000 * np.linalg.norm(self.reciprocity_difference_mv, 2), 1)

        # plot
        if verbose:
            # pulse
            # filter
            filter = self.add_conj(self.filter)
            pulse_response = self.pulse * filter
            v_filtered = np.real(np.fft.ifft(pulse_response))
            ax = axs[1, 0]
            if self.pulse_shape == 1:
                ax.plot(self.t_pulse, self.v_ref, color = 'r',
                        label = 'Reference')
                ax.plot(self.t_pulse, self.v_pulse, color = 'k', linestyle = 'dashed',
                        label = 'Generated')
                ax.legend(loc = 'upper right')
                ax.set_title('Gaussian Pulse')
            else:
                ax.plot(self.t_ref, self.v_ref, color = 'r', marker = 'o',
                        label = 'Reference')
                ax.plot(self.t_pulse, self.v_pulse, color = 'k', linestyle = 'dashed',
                        label = 'Interpolated')
                ax.plot(self.t_pulse, v_filtered, color = 'b', linestyle = 'dotted',
                        label = 'Filtered')
                ax.set_title('Filtered Rectangular Pulse')
            ax.legend(loc = 'upper right')
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')


            # time domain transmission
            ax = axs[0, 1]
            ax.set_title('TDR11')
            ax.plot(self.t_causal * 1e9, self.v_causal[:, 0, 0] / 2.,
                    label = 'causal', color = 'r')
            ax.plot(self.t_origin * 1e9, self.v_origin[:, 0, 0] / 2.,
                    label = 'original', color = 'k', linestyle = 'dashed')
            ax = axs[0, 2]
            ax.set_title('TDT21')
            ax.plot(self.t_causal * 1e9, self.v_causal[:, 1, 0] / 2.,
                    label = 'causal', color = 'r')
            ax.plot(self.t_origin * 1e9, self.v_origin[:, 1, 0] / 2.,
                    label = 'original', color = 'k', linestyle = 'dashed')
            ax = axs[1, 1]
            ax.set_title('TDT12')
            ax.plot(self.t_causal * 1e9, self.v_causal[:, 0, 1] / 2.,
                    label = 'causal', color = 'r')
            ax.plot(self.t_origin * 1e9, self.v_origin[:, 0, 1] / 2.,
                    label = 'original', color = 'k', linestyle = 'dashed')
            ax = axs[1, 2]
            ax.set_title('TDR22')
            ax.plot(self.t_causal * 1e9, self.v_causal[:, 1, 1] / 2.,
                    label = 'causal', color = 'r')
            ax.plot(self.t_origin * 1e9, self.v_origin[:, 1, 1] / 2.,
                    label = 'original', color = 'k', linestyle = 'dashed')
            for ax in axs[:, 1:].reshape(-1):
                ax.set_xlabel('Time (ns)')
                ax.set_ylabel('Amplitude (V)')
                ax.legend(loc = 'upper right')
            fig.tight_layout()

        QM = {'causality': {'value': causality_metric / 2., 'unit': 'mV',
                            'evaluation': ''},
             'passivity': {'value': passivity_metric / 2., 'unit': 'mV',
                           'evaluation': ''},
              'reciprocity': {'value': reciprocity_metric / 2., 'unit': 'mV',
                              'evaluation': ''},
              }

        # evaluation
        CQM = QM['causality']['value']
        if CQM >= 15.:
            QM['causality']['evaluation'] = 'poor'
        elif CQM >= 10.:
            QM['causality']['evaluation'] = 'inconclusive'
        elif CQM >= 5.:
            QM['causality']['evaluation'] = 'acceptable'
        else:
            QM['causality']['evaluation'] = 'good'

        PQM = QM['passivity']['value']
        if PQM >= 15.:
            QM['passivity']['evaluation'] = 'poor'
        elif PQM >= 10.:
            QM['passivity']['evaluation'] = 'inconclusive'
        elif PQM >= 5.:
            QM['passivity']['evaluation'] = 'acceptable'
        else:
            QM['passivity']['evaluation'] = 'good'

        RQM = QM['reciprocity']['value']
        if RQM >= 15.:
            QM['reciprocity']['evaluation'] = 'poor'
        elif RQM >= 10.:
            QM['reciprocity']['evaluation'] = 'inconclusive'
        elif RQM >= 5.:
            QM['reciprocity']['evaluation'] = 'acceptable'
        else:
            QM['reciprocity']['evaluation'] = 'good'

        return QM

    def check_mm_quality(self, ntwk: Network, verbose: bool = False) -> dict:
        """
        Mixed-mode application-based quality checking of in the time domain.

        If necessary, the original S-parameters are extrapolated to a frequency
        of three times the desired data rate. Causal, passive, and reciprocal
        models are reconstructed and stimulated by a pulse. The difference
        between the original extrapolated signal and the ideal responses
        in the time domain are integrated to give metrics in millivolts.

        The input networks should be 4-port single-ended and will be
        transformed to mixed-mode representation. Only the differential and
        the common modes are tested.

        Parameters
        ----------
        ntwk:    :class:`~skrf.network.Network` object
                 Network to be checked
        verbose: :bool
                 Plot internal causality, passivity, and reciprocity
                 figures. When True, override class verbose parameter.
                 (default False).

        Returns
        -------
        QM : :class:`dict` object
              Dictionary with quality metrics
        """
        mm = ntwk.copy()
        mm.se2gmm(p = 2)
        QM = {'dd': self.check_se_quality(mm.subnetwork([0, 1]), verbose),
              'cc': self.check_se_quality(mm.subnetwork([2, 3]), verbose)}

        return QM

    def print_qm(self, QM: dict) -> dict:
        """
        Print the quality metrics dictionary.

        Parameters
        ----------
        QM: :class:`dict` object
            Dictionary with quality metrics to print
        """
        # Ignoring T201 for QM printout as it's not a user-facing print statement
        if 'dd' in QM:
            print('Differential mode') # noqa: T201
            for k in QM['dd'].keys():
                print(f"{k} in the time domain is {QM['dd'][k]['evaluation']} " # noqa: T201
                       f"({QM['dd'][k]['value']} {QM['dd'][k]['unit']})")
            print('Common mode') # noqa: T201
            for k in QM['cc'].keys():
                print(f"{k} in the time domain is {QM['cc'][k]['evaluation']} " # noqa: T201
                       f"({QM['cc'][k]['value']} {QM['cc'][k]['unit']})")
        else:
            for k in QM.keys():
                print(f"{k} in the time domain is {QM[k]['evaluation']} " # noqa: T201
                       f"({QM[k]['value']} {QM[k]['unit']})")

class IEEEP370_SE_NZC_2xThru(IEEEP370):
    """
    Creates error boxes from a test fixture 2xThru network.

    Based on [ElSA20]_ and [I3E370]_.

    A deembedding object is created with a single 2xThru (FIX-FIX) network,
    which is split into left (FIX-1) and right (FIX-2) fixtures with IEEEP370
    2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1 and
    FIX-2 are deembedded from the FIX_DUT_FIX network.

    This method is applicable only when there is a 2x-Thru network.

    The S-parameters bisection is done by time gating S11 and S22, taking the
    proper square root of the S21 corrected by return loss, and remixing the
    parameters according to the fixture signal flow graph. This method gives
    crude results but is robust.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import IEEEP370_SE_NZC_2xThru

    Create network objects for 2x-Thru and FIX_DUT_FIX

    >>> s2xthru = rf.Network('2xthru.s2p')
    >>> fdf = rf.Network('f-dut-f.s2p')

    Create de-embedding object

    >>> dm = IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru, name = '2xthru')

    Apply deembedding to get the actual DUT network

    >>> dut = dm.deembed(fdf)

    Note
    ----
    numbering diagram::

         FIX-1    DUT      FIX-2
         +----+   +----+   +----+
        -|1  2|---|1  2|---|2  1|-
         +----+   +----+   +----+


    Warning
    -------
    There are two differences compared to the original matlab implementation
    [I3E370]:
        - FIX-2 is flipped (see diagram above)
        - A more robust root choice solution is used that avoids the apparition
          of 180° phase jumps in the fixtures in certain circumstances

    References
    ----------
    .. [ElSA20] Ellison J, Smith SB, Agili S., "Using a 2x-thru standard to achieve
        accurate de-embedding of measurements", Microwave Optical Technology
        Letter, 2020, https://doi.org/10.1002/mop.32098
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP3702xThru.m,
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    """
    def __init__(self, dummy_2xthru: Network, name: str = None,
                 z0: float = 50, use_z_instead_ifft: bool = False, verbose: bool = False,
                 forced_z0_line: float = None, *args, **kwargs) -> None:
        """
        IEEEP370_SE_NZC_2xThru De-embedding Initializer

        Parameters
        -----------

        dummy_2xthru : :class:`~skrf.network.Network` object
            2xThru (FIX-FIX) network.

        z0 :
            reference impedance of the S-parameters (default: 50)

        name : string
            optional name of de-embedding object

        use_z_instead_ifft:
            use z-transform instead ifft. This method is not documented in
            the paper but exists in the IEEE repo. It could be used if the
            2x-Thru is so short that there is not enough points in time domain
            to determine the length of half fixtures from the s21 impulse
            response and the the impedance at split plane from the s11 step
            response.
            Parameter `verbose` could be used for diagnostic in
            ifft mode. (default: False)

        forced_z0_line:
            If specified, the value for the split plane impedance is forced to
            `forced_z0_line`.
            The IEEEP370 standard recommends the 2x-Thru being at least three
            wavelengths at the highest measured frequency. This ensures that
            the split plane impedance measured in the S11 step response is free
            of reflections from the launches.
            If the 2x-Thru is too short, any point in the s11 step response
            contain reflections from the lanches and split plane impedance
            cannot be determined accurately by this method.
            In this case, setting the impedance manually can improve the
            results. However, it should be noted that each fixture model will
            still include some reflections from the opposite side launch
            because there is not enough time resolution to separate them.
            (Default: None)

        verbose :
            view the process (default: False)

        args, kwargs:
            passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.s2xthru = dummy_2xthru.copy()
        self.z0 = z0
        dummies = [self.s2xthru]
        self.use_z_instead_ifft = use_z_instead_ifft
        self.forced_z0_line = forced_z0_line
        self.verbose = verbose
        # debug outputs
        self.x_end = None
        self.z_x = None

        IEEEP370.__init__(self, dummies, name, *args, **kwargs)
        self.s_side1, self.s_side2 = self.split2xthru(self.s2xthru)

    def deembed(self, ntwk: Network) -> Network:
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            FIX-DUT-FIX network from which FIX-1 AND FIX-2 fixtures needs to be
            removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.s2xthru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            s_side1, s_side2 = self.split2xthru(s2xthru)
        else:
            s_side1 = self.s_side1
            s_side2 = self.s_side2

        return s_side1.inv ** ntwk ** s_side2.flipped().inv

    def split2xthru(self, s2xthru: Network) -> (Network, Network):
        """
        Perform the fixtures extraction.
        """
        f = s2xthru.frequency.f
        s = s2xthru.s

        if not self.use_z_instead_ifft:
            # strip DC point if one exists
            if(f[0] == 0):
                warnings.warn(
                    "DC point detected. An interpolated DC point will be included in the errorboxes.",
                    RuntimeWarning, stacklevel=2
                    )
                flag_DC = True
                f = f[1:]
                s = s[1:]
            else:
                flag_DC = False

            # interpolate S-parameters if the frequency vector is not acceptable
            if(f[1] - f[0] != f[0]):
                warnings.warn(
                   """Non-uniform frequency vector detected. An interpolated S-parameter matrix will be created for
                   this calculation. The output results will be re-interpolated to the original vector.""",
                   RuntimeWarning, stacklevel=2
                   )
                flag_df = True
                f_original = f
                projected_n = round(f[-1]/f[0])
                if(projected_n <= 10000):
                    fnew = f[0] * (np.arange(0, projected_n) + 1)
                else:
                    dfnew = f[-1]/10000
                    fnew = dfnew * (np.arange(0, 10000) + 1)
                stemp = Network(frequency = Frequency.from_f(f, 'Hz'), s = s)
                f_interp = Frequency.from_f(fnew, unit = 'Hz')
                stemp.interpolate_self(f_interp, kind = 'cubic',
                                       fill_value = 'extrapolate')
                f = fnew
                s = stemp.s
                del stemp

            else:
                flag_df = False

            n = len(f)
            s11 = s[:, 0, 0]

            # get e001 and e002
            # e001
            s21 = s[:, 1, 0]
            dcs21 = IEEEP370.dc_interp(s21, f)
            t21 = fftshift(irfft(concatenate(([dcs21], s21)), axis=0), axes=0)
            x = np.argmax(t21)
            self.x_end = x

            dcs11 = IEEEP370.DC(s11,f)
            t11 = fftshift(irfft(concatenate(([dcs11], s11)), axis=0), axes=0)
            step11 = IEEEP370.makeStep(t11)
            z11 = -self.z0 * (step11 + 1) / (step11 - 1)

            if self.forced_z0_line:
                z11x = self.forced_z0_line
            else:
                z11x = 0.5 * (z11[x-1] + z11[x])
            self.z_x = z11x

            if self.verbose:
                fig, (ax1, ax2) = subplots(2,1, sharex = True)
                fig.suptitle('Midpoint length and impedance determination')
                ax1.plot(t21, label = 't21')
                ax1.plot([x], [t21[x]], marker = 'o', linestyle = 'none',
                            label = 't21x')
                ax1.grid()
                ax1.legend()
                ax2.plot(z11, label = 'z11')
                ax2.plot([x], [z11x], marker = 'o', linestyle = 'none',
                            label = 'z11x')
                ax2.set_xlabel('t-samples')
                ax2.set_xlim((x - 50, x + 50))
                ax2.grid()
                ax2.legend()

            temp = Network(frequency = Frequency.from_f(f, 'Hz'), s = s, z0 = self.z0)
            temp.renormalize(z11x)
            sr = temp.s
            del temp

            s11r = sr[:, 0, 0]
            s21r = sr[:, 1, 0]
            s12r = sr[:, 0, 1]
            s22r = sr[:, 1, 1]

            dcs11r = IEEEP370.DC(s11r, f)
            # irfft is equivalent to ifft(makeSymmetric(x))
            t11r = fftshift(irfft(concatenate(([dcs11r], s11r)), axis=0), axes=0)
            t11r[x:] = 0
            e001 = fft(ifftshift(t11r))
            e001 = e001[1:n+1]

            dcs22r = IEEEP370.DC(s22r, f)
            t22r = fftshift(irfft(concatenate(([dcs22r], s22r)), axis=0), axes=0)
            t22r[x:] = 0
            e002 = fft(ifftshift(t22r))
            e002 = e002[1:n+1]

            # calc e111 and e112
            e111 = (s22r - e002) / s12r
            e112 = (s11r - e001) / s21r

            # original implementation, 180° phase jumps in case of phase noise
            # # calc e01
            # k = 1
            # test = k * np.sqrt(s21r * (1 - e111 * e112))
            # e01 = zeros((n), dtype = complex)
            # for i, value in enumerate(test):
            #     if(i>0):
            #         if(angle(test[i]) - angle(test[i-1]) > 0):
            #             k = -1 * k
            #     # mhuser : is it a problem with complex value cast to real here ?
            #     e01[i] = k * np.sqrt(s21r[i] * (1 - e111[i] * e112[i]))

            # # calc e10
            # k = 1
            # test = k * np.sqrt(s12r * (1 - e111 * e112))
            # e10 = zeros((n), dtype = complex)
            # for i, value in enumerate(test):
            #     if(i>0):
            #         if(angle(test[i]) - angle(test[i-1]) > 0):
            #             k = -1 * k
            #     # mhuser : is it a problem with complex value cast to real here ?
            #     e10[i] = k * np.sqrt(s12r[i] * (1 - e111[i] * e112[i]))

            # calc e01 and e10
            # avoid 180° phase jumps in case of phase noise
            e01 = np.sqrt(s21r * (1 - e111 * e112))
            for i in range(n):
                if i > 0:
                    if np.abs(-e01[i] - e01[i-1]) < np.abs(e01[i] - e01[i-1]):
                        e01[i] = - e01[i]
            e10 = np.sqrt(s12r * (1 - e111 * e112))
            for i in range(n):
                if i > 0:
                    if np.abs(-e10[i] - e10[i-1]) < np.abs(e10[i] - e10[i-1]):
                        e10[i] = - e10[i]


            # revert to initial freq axis
            if flag_df:
                interp1d = scipy.interpolate.interp1d
                interp_e001 = interp1d(f, e001, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e001 = interp_e001(f_original)
                interp_e01 = interp1d(f, e01, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e01 = interp_e01(f_original)
                interp_e111 = interp1d(f, e111, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e111 = interp_e111(f_original)
                interp_e002 = interp1d(f, e002, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e002 = interp_e002(f_original)
                interp_e10 = interp1d(f, e10, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e10 = interp_e10(f_original)
                interp_e112 = interp1d(f, e112, kind = 'cubic',
                                fill_value = 'extrapolate',
                                assume_sorted = True)
                e112 = interp_e112(f_original)
                f = f_original

            # dc point was included in the original file
            if flag_DC:
                e001 = concatenate(([IEEEP370.dc_interp(e001, f)], e001))
                e01  = concatenate(([IEEEP370.dc_interp(e01, f)], e01))
                e111 = concatenate(([IEEEP370.dc_interp(e111, f)], e111))
                e002 = concatenate(([IEEEP370.dc_interp(e002, f)], e002))
                e10 = concatenate(([IEEEP370.dc_interp(e10, f)], e10))
                e112 = concatenate(([IEEEP370.dc_interp(e112, f)], e112))
                f = concatenate(([0], f))

            # S-parameters are now setup correctly
            n = len(f)
            fixture_model_1r = zeros((n, 2, 2), dtype = complex)
            fixture_model_1r[:, 0, 0] = e001
            fixture_model_1r[:, 1, 0] = e01
            fixture_model_1r[:, 0, 1] = e01
            fixture_model_1r[:, 1, 1] = e111

            fixture_model_2r = zeros((n, 2, 2), dtype = complex)
            fixture_model_2r[:, 1, 1] = e002
            fixture_model_2r[:, 0, 1] = e10
            fixture_model_2r[:, 1, 0] = e10
            fixture_model_2r[:, 0, 0] = e112

            # create the S-parameter objects for the errorboxes
            s_fixture_model_r1  = Network(frequency = Frequency.from_f(f, 'Hz'), s = fixture_model_1r, z0 = z11x)
            s_fixture_model_r2  = Network(frequency = Frequency.from_f(f, 'Hz'), s = fixture_model_2r, z0 = z11x)

            # renormalize the S-parameter errorboxes to the original reference impedance
            s_fixture_model_r1.renormalize(self.z0)
            s_fixture_model_r2.renormalize(self.z0)
            s_side1 = s_fixture_model_r1
            s_side2 = s_fixture_model_r2.flipped() # FIX-2 is flipped in skrf

        else:
            z = s2xthru.z
            ZL = zeros(z.shape, dtype = complex)
            ZR = zeros(z.shape, dtype = complex)

            for i in range(len(f)):
                ZL[i, :, :] = [
                    [z[i, 0, 0] + z[i, 1, 0], 2. * z[i, 1, 0]],
                    [2. * z[i, 1, 0], 2. * z[i, 1, 0]]
                              ]
                ZR[i, :, :] = [
                    [2. * z[i, 0, 1], 2. * z[i, 0, 1]],
                    [2. * z[i, 0, 1], z[i, 1, 1] + z[i, 0, 1]]
                              ]

            s_side1 = Network(frequency = s2xthru.frequency, z = ZL, z0 = self.z0)
            s_side2 = Network(frequency = s2xthru.frequency, z = ZR, z0 = self.z0)
            s_side2.flip() # FIX-2 is flipped in skrf

        return (s_side1, s_side2)

    def plot_check_residuals(self, ax: Axes = None) -> (Figure, Axes):
        res = self.deembed(self.s2xthru)
        res.name = 'Residuals'

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #1: Self de-embedding of 2X-Thru')

        ax[0].set_title('Magnitude residuals')
        res.plot_s_db(1,0, ax = ax[0], color = '0.5')
        res.plot_s_db(0,1, ax = ax[0], color = 'k')
        ax[0].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [0.1, 0.1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[0].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [-0.1, -0.1],
                       linestyle = 'dashed', color = 'r')
        ax[0].legend(loc = 'upper right')

        ax[1].set_title('Phase residuals')
        res.plot_s_deg(1,0, ax = ax[1], color = '0.5')
        res.plot_s_deg(0,1, ax = ax[1], color = 'k')
        ax[1].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [1, 1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[1].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [-1, -1],
                       linestyle = 'dashed', color = 'r')
        ax[1].legend(loc = 'upper right')
        fig.tight_layout()

        return (fig, ax)

    def plot_check_impedance(self, fix_dut_fix: Network = None, ax: Axes = None,
                             window: str = 'hamming') -> (Figure, Axes):
        # if dc point already exists, it will be replaced
        s2xthru = IEEEP370.extrapolate_to_dc(self.s2xthru)
        fix1 = IEEEP370.extrapolate_to_dc(self.s_side1)
        fix2 = IEEEP370.extrapolate_to_dc(self.s_side2)
        if fix_dut_fix is not None:
            fix_dut_fix = IEEEP370.extrapolate_to_dc(fix_dut_fix)
        n = s2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * s2xthru.frequency.step) # ns

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #2: Compare the TDR of the fixture model to the FIX-DUT-FIX')
        ax[0].set_title('Side 1')
        fix1.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], color = 'k')
        s2xthru.plot_z_time_step(0, 0, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        y = ax[0].lines[-1].get_ydata()
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(0, 0, window = window,
                                     ax = ax[0], linestyle = 'dashed', color = 'm')
        ax[0].plot([0], [y[n // 2]], marker = 's', color = 'k', label = 'start')
        ax[0].plot([(self.x_end - (n // 2) - 1) * dt], [self.z_x], marker = 'o',
                   color = 'k', label = f'z_x = {self.z_x:0.1f} ohm')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax[0].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[0].lines[1].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ymin = np.min(np.array([ax[0].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[0].lines[1].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ax[0].set_ylim((ymin - 5, ymax + 5))
        ax[0].legend(loc = 'lower left')

        ax[1].set_title('Side 2')
        fix2.plot_z_time_step(0, 0, window = window,
                              ax = ax[1], color = 'k')
        s2xthru.plot_z_time_step(1, 1, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        y = ax[1].lines[-1].get_ydata()
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(1, 1, window = window,
                                     ax = ax[1], linestyle = 'dashed', color = 'm')
        ax[1].plot([0], [y[n // 2]], marker = 's', color = 'k', label = 'start')
        ax[1].plot([(self.x_end - (n // 2) - 1) * dt], [self.z_x], marker = 'o',
                   color = 'k', label = f'z_x = {self.z_x:0.1f} ohm')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax[1].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[1].lines[1].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ymin = np.min(np.array([ax[1].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[1].lines[1].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ax[1].set_ylim((ymin - 5, ymax + 5))
        delay = 2 * (self.x_end - (n // 2)) * dt
        ax[1].set_xlim((-0.5 * delay, 1.5 * delay))
        ax[1].legend(loc = 'lower left')

        fig.tight_layout()

        return (fig, ax)

class IEEEP370_MM_NZC_2xThru(IEEEP370):
    """
    Creates error boxes from a 4-port test fixture 2xThru.

    Based on [ElSA20]_ and [I3E370]_.

    A deembedding object is created with a single 2xThru (FIX-FIX) network,
    which is split into left (FIX-1_2) and right (FIX-3_4) fixtures with
    IEEEP370 2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1 and
    FIX-2 are deembedded from the FIX_DUT_FIX network.

    This method is applicable only when there is a 2x-Thru measurement.

    The S-parameters bisection is done by time gating S11 and S22, taking the
    proper square root of the S21 corrected by return loss, and remixing the
    parameters according to the fixture signal flow graph. This method gives
    crude results but is robust.

    Note
    ----
    The `port_order` ='first', means front-to-back also known as odd/even,
    while `port_order`='second' means left-to-right also known as sequential.
    `port_order`='third' means yet another numbering method.
    Next figure show example of port numbering with 4-port networks.

    The `scikit-rf` cascade ** 2N-port operator use second scheme. This is very
    convenient to write compact deembedding and other expressions.

    numbering diagram::

      port_order = 'first'
           +---------+
          -|0       1|-
          -|2       3|-
           +---------+

      port_order = 'second'
           +---------+
          -|0       2|-
          -|1       3|-
           +---------+

      port_order = 'third'
           +---------+
          -|0       3|-
          -|1       2|-
           +---------+


    use `Network.renumber` to change port ordering.

    Example
    -------
    >>> import skrf as rf
    >>> from skrf.calibration import IEEEP370_MM_NZC_2xThru

    Create network objects for 2xThru and FIX-DUT-FIX

    >>> s2xthru = rf.Network('2xthru.s4p')
    >>> fdf = rf.Network('f-dut-f.s4p')

    Create de-embedding object

    >>> dm = IEEEP370_MM_NZC_2xThru(dummy_2xthru = s2xthru, name = '2xthru')

    Apply deembedding to get the actual DUT network

    >>> dut = dm.deembed(fdf)

    Note
    ----
    numbering diagram::

         FIX-1_2  DUT      FIX-3_4
         +----+   +----+   +----+
        -|1  3|---|1  3|---|3  1|-
        -|2  4|---|2  4|---|4  2|-
         +----+   +----+   +----+


    Warning
    -------
    There are two differences compared to the original matlab implementation
    [I3E370]_:
    - FIX-2 is flipped (see diagram above)
    - A more robust root choice solution is used that avoids the apparition
    of 180° phase jumps in the fixtures in certain circumstances

    References
    ----------
    .. [ElSA20] Ellison J, Smith SB, Agili S., "Using a 2x-thru standard to achieve
        accurate de-embedding of measurements", Microwave Optical Technology
        Letter, 2020, https://doi.org/10.1002/mop.32098
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370mmZc2xthru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    """
    def __init__(self, dummy_2xthru: Network, name: str = None,
                 z0: float = 50, port_order: PortOrderT = 'second',
                 use_z_instead_ifft: bool = False, verbose: bool = False,
                 forced_z0_line_dd: float = None, forced_z0_line_cc: float = None,
                 *args, **kwargs) -> None:
        """
        IEEEP370_MM_NZC_2xThru De-embedding Initializer

        Parameters
        -----------

        dummy_2xthru : :class:`~skrf.network.Network` object
            2xThru (FIX-FIX) network.

        z0 :
            reference impedance of the S-parameters (default: 50)

        port_order : ['first', 'second', 'third']
            specify what numbering scheme to use. See above. (default: second)

        name : string
            optional name of de-embedding object

        use_z_instead_ifft:
            use z-transform instead ifft. This method is not documented in
            the paper but exists in the IEEE repo. It could be used if the
            2x-Thru is so short that there is not enough points in time domain
            to determine the length of half fixtures from the s21 impulse
            response and the the impedance at split plane from the s11 step
            response.
            Parameter `verbose` could be used for diagnostic in
            ifft mode. (default: False)

        forced_z0_line_dd:
            If specified, the value for the split plane impedance is forced to
            `forced_z0_line` for differential-mode.
            The IEEEP370 standard recommends the 2x-Thru being at least three
            wavelengths at the highest measured frequency. This ensures that
            the split plane impedance measured in the S11 step response is free
            of reflections from the launches.
            If the 2x-Thru is too short, any point in the s11 step response
            contain reflections from the lanches and split plane impedance
            cannot be determined accurately by this method.
            In this case, setting the impedance manually can improve the
            results. However, it should be noted that each fixture model will
            still include some reflections from the opposite side launch
            because there is not enough time resolution to separate them.
            (Default: None)

        forced_z0_line_cc:
            Same behaviour as `forced_z0_line_dd`, but for the common-mode
            split plane impedance.
            (Default: None)

        verbose :
            view the process (default: False)

        args, kwargs:
            passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.s2xthru = dummy_2xthru.copy()
        self.z0 = z0
        self.port_order = port_order
        dummies = [self.s2xthru]
        self.use_z_instead_ifft = use_z_instead_ifft
        self.verbose = verbose
        self.forced_z0_line_dd = forced_z0_line_dd
        self.forced_z0_line_cc = forced_z0_line_cc
        # debug outputs
        self.x_end_dd = None
        self.z_x_dd = None
        self.x_end_cc = None
        self.z_x_cc = None

        IEEEP370.__init__(self, dummies, name, *args, **kwargs)
        self.se_side1, self.se_side2 = self.split2xthru(self.s2xthru)

    def deembed(self, ntwk: Network) -> Network:
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            FIX-DUT-FIX network from which FIX-1_2 and FIX-3_4 fixtures needs
            to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.s2xthru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            se_side1, se_side2 = self.split2xthru(s2xthru)
        else:
            se_side1 = self.se_side1
            se_side2 = self.se_side2

        # check if 4-port
        if ntwk.nports != 4:
            raise(ValueError('2xthru has to be a 4-port network.'))
        # renumber if required
        if self.port_order == 'first':
            N = ntwk.nports
            old_order = list(range(N))
            new_order = list(range(0, N, 2)) + list(range(1, N, 2))
            ntwk.renumber(old_order, new_order)
        elif self.port_order == 'third':
            N = ntwk.nports
            old_order = list(range(N))
            new_order = list(range(0, N//2)) + list(range(N-1, N//2-1, -1))
            ntwk.renumber(old_order, new_order)

        deembedded = se_side1.inv ** ntwk ** se_side2.flipped().inv
        #renumber back if required
        if self.port_order != 'second':
            deembedded.renumber(new_order, old_order)
        return deembedded

    def split2xthru(self, se_2xthru: Network) -> (Network, Network):
        """
        Perform the fixtures extraction.
        """
        # check if 4-port
        if se_2xthru.nports != 4:
            raise(ValueError('2xthru has to be a 4-port network.'))
        # renumber if required
        if self.port_order == 'first':
            N = se_2xthru.nports
            old_order = list(range(N))
            new_order = list(range(0, N, 2)) + list(range(1, N, 2))
            se_2xthru.renumber(old_order, new_order)
        elif self.port_order == 'third':
            N = se_2xthru.nports
            old_order = list(range(N))
            new_order = list(range(0, N//2)) + list(range(N-1, N//2-1, -1))
            se_2xthru.renumber(old_order, new_order)

        #convert to mixed-modes
        mm_2xthru = se_2xthru.copy()
        mm_2xthru.se2gmm(p = 2)

        #extract common and differential mode and model fixtures for each
        sdd = subnetwork(mm_2xthru, [0, 1])
        scc = subnetwork(mm_2xthru, [2, 3])
        dm_dd  = IEEEP370_SE_NZC_2xThru(dummy_2xthru = sdd, z0 = self.z0 * 2,
                                use_z_instead_ifft = self.use_z_instead_ifft,
                                verbose = self.verbose,
                                forced_z0_line = self.forced_z0_line_dd)
        self.x_end_dd = dm_dd.x_end
        self.z_x_dd = dm_dd.z_x

        dm_cc  = IEEEP370_SE_NZC_2xThru(dummy_2xthru = scc, z0 = self.z0 / 2,
                                use_z_instead_ifft = self.use_z_instead_ifft,
                                verbose = self.verbose,
                                forced_z0_line = self.forced_z0_line_cc)
        self.x_end_cc = dm_cc.x_end
        self.z_x_cc = dm_cc.z_x

        #convert back to single-ended
        mm_side1 = concat_ports([dm_dd.s_side1, dm_cc.s_side1], port_order = 'first')
        se_side1 = mm_side1.copy()
        se_side1.gmm2se(p = 2)
        mm_side2 = concat_ports([dm_dd.s_side2, dm_cc.s_side2], port_order = 'first')
        se_side2 = mm_side2.copy()
        se_side2.gmm2se(p = 2)

        return (se_side1, se_side2)

    def plot_check_residuals(self, ax: Axes = None) -> (Figure, Axes):
        res = self.deembed(self.s2xthru)
        res.name = 'Residuals'
        res.se2gmm(p=2)

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #1: Self de-embedding of 2X-Thru')

        ax[0].set_title('Magnitude residuals')
        res.plot_s_db(1,0, ax = ax[0])
        res.plot_s_db(0,1, ax = ax[0])
        res.plot_s_db(3,2, ax = ax[0])
        res.plot_s_db(2,3, ax = ax[0])
        ax[0].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [0.1, 0.1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[0].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [-0.1, -0.1],
                       linestyle = 'dashed', color = 'r')
        ax[0].legend(loc = 'upper right')

        ax[1].set_title('Phase residuals')
        res.plot_s_deg(1,0, ax = ax[1])
        res.plot_s_deg(0,1, ax = ax[1])
        res.plot_s_deg(3,2, ax = ax[1])
        res.plot_s_deg(2,3, ax = ax[1])
        ax[1].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [1, 1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[1].plot([res.frequency.f_scaled[0], res.frequency.f_scaled[-1]],
                       [-1, -1],
                       linestyle = 'dashed', color = 'r')
        ax[1].legend(loc = 'upper right')
        fig.tight_layout()

        return (fig, ax)

    def plot_check_impedance(self, fix_dut_fix: Network = None, ax: Axes = None,
                             window: str = 'hamming') -> (Figure, Axes):
        # if dc point already exists, it will be replaced
        s2xthru = self.s2xthru.copy()
        s2xthru.se2gmm(p=2)
        s2xthru = IEEEP370.extrapolate_to_dc(s2xthru)
        fix1 = self.se_side1.copy()
        fix1.se2gmm(p=2)
        fix1 = IEEEP370.extrapolate_to_dc(fix1)
        fix2 = self.se_side2.copy()
        fix2.se2gmm(p=2)
        fix2 = IEEEP370.extrapolate_to_dc(fix2)
        if fix_dut_fix is not None:
            fix_dut_fix = fix_dut_fix.copy()
            fix_dut_fix.se2gmm(p=2)
            fix_dut_fix = IEEEP370.extrapolate_to_dc(fix_dut_fix)
        n = s2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * s2xthru.frequency.step) # ns

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #2: Compare the TDR of the fixture model to the FIX-DUT-FIX')
        ax[0].set_title('Side 1')
        fix1.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], color = 'k')
        s2xthru.plot_z_time_step(0, 0, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        y = ax[0].lines[-1].get_ydata()
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(0, 0, window = window,
                                     ax = ax[0], linestyle = 'dashed', color = 'm')
        ax[0].plot([0], [y[n // 2]], marker = 's', color = 'k', label = 'start')
        ax[0].plot([(self.x_end_dd - (n // 2) - 1) * dt], [self.z_x_dd], marker = 'o',
                   color = 'k', label = f'z_x = {self.z_x_dd:0.1f} ohm')
        fix1.plot_z_time_step(2, 2, window = window,
                              ax = ax[0], color = 'k')
        s2xthru.plot_z_time_step(2, 2, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(2, 2, window = window,
                                     ax = ax[0], linestyle = 'dashed', color = 'b')
        ax[0].legend(loc = 'center left')

        ax[1].set_title('Side 2')
        fix2.plot_z_time_step(0, 0, window = window,
                              ax = ax[1], color = 'k')
        s2xthru.plot_z_time_step(1, 1, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        y = ax[1].lines[-1].get_ydata()
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(1, 1, window = window,
                                     ax = ax[1], linestyle = 'dashed', color = 'm')
        ax[1].plot([0], [y[n // 2]], marker = 's', color = 'k', label = 'start')
        ax[1].plot([(self.x_end_dd - (n // 2) - 1) * dt], [self.z_x_dd], marker = 'o',
                   color = 'k', label = f'z_x = {self.z_x_dd:0.1f} ohm')
        fix2.plot_z_time_step(2, 2, window = window,
                              ax = ax[1], color = 'k')
        s2xthru.plot_z_time_step(3, 3, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        if fix_dut_fix is not None:
            fix_dut_fix.plot_z_time_step(3, 3, window = window,
                                     ax = ax[1], linestyle = 'dashed', color = 'b')
        delay = 2 * (self.x_end_dd - (n // 2)) * dt
        ax[1].set_xlim((-0.5 * delay, 1.5 * delay))
        ax[1].legend(loc = 'center left')

        fig.tight_layout()

        return (fig, ax)


class IEEEP370_SE_ZC_2xThru(IEEEP370):
    """
    Creates error boxes from 2x-Thru and FIX-DUT-FIX networks.

    Based on [I3E370]_.

    A deembedding object is created with 2x-Thru (FIX-FIX) and FIX-DUT-FIX
    measurements, which are split into left (FIX-1) and right (FIX-2) fixtures
    with IEEEP370 Zc2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1 and
    FIX-2 are deembedded from FIX_DUT_FIX network.

    This method is applicable only when there is 2xThru and FIX_DUT_FIX
    networks.

    The possible difference of impedance between 2x-Thru and FIX-DUT-FIX
    is corrected.

    The algorithm computes the length of the fixtures by halving the delay of
    2x-Thru in time domain transmission. The propagation constant gamma is also
    determined from the 2xThru. It then peels the FIX-DUT-FIX time domain
    impedance profile iteratively in cycles of determining start impedance and
    deembedding a single time sample long transmission line.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import IEEEP370_SE_ZC_2xThru

    Create network objects for 2xThru and FIX-DUT-FIX

    >>> s2xthru = rf.Network('2xthru.s2p')
    >>> fdf = rf.Network('f-dut-f.s2p')

    Create de-embedding object

    >>> dm = IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru, dummy_fix_dut_fix = fdf,
                             bandwidth_limit = 10e9,
                             pullback1 = 0, pullback2 = 0,
                             leadin = 0,
                             name = 'zc2xthru')

    Apply deembedding to get the DUT

    >>> dut = dm.deembed(fdf)

    Note
    ----
    numbering diagram::

         FIX-1    DUT      FIX-2
         +----+   +----+   +----+
        -|1  2|---|1  2|---|2  1|-
         +----+   +----+   +----+


    Warning
    -------
    There is one difference compared to the original matlab implementation
    [I3E370]_:
    - FIX-2 is flipped (see diagram above)

    References
    ----------
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370Zc2xThru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    """
    def __init__(self, dummy_2xthru: Network, dummy_fix_dut_fix: Network,
                 name: str = None,
                 z0: float = 50, bandwidth_limit: float = 0,
                 pullback1: int = 0, pullback2: int = 0,
                 side1: bool = True, side2: bool = True,
                 NRP_enable: bool = True, leadin: int = 1,
                 verbose: bool = False,
                 *args, **kwargs) -> None:
        """
        IEEEP370_SE_ZC_2xThru De-embedding Initializer

        Parameters
        -----------

        dummy_2xthru : :class:`~skrf.network.Network` object
            2xThru (FIX-FIX) network.

        name : string
            optional name of de-embedding object

        z0 :
            reference impedance of the S-parameters (default: 50)

        bandwidth_limit :
            max frequency for a fitting function
            (default: 0, use all s-parameters without fit)

        pullback1, pullback2 :
            a number of discrete points to leave in the fixture on side 1
            respectively on side 2 (default: 0 leave all)

        side1, side2 :
            set to de-embed the side1 resp. side2 errorbox (default: True)

        NRP_enable :
            set to enforce the Nyquist Rate Point during de-embedding and to
            add the appropriote delay to the errorboxes (default: True)

        leadin :
            a number of discrete points before t = 0 that are non-zero from
            calibration error (default: 1)

        verbose :
            view the process (default: False)

        args, kwargs:
            passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.s2xthru = dummy_2xthru.copy()
        self.sfix_dut_fix = dummy_fix_dut_fix.copy()
        dummies = [self.s2xthru]
        self.z0 = z0
        self.bandwidth_limit = bandwidth_limit
        self.pullback1 = pullback1
        self.pullback2 = pullback2
        self.side1 = side1
        self.side2 = side2
        self.NRP_enable = NRP_enable
        self.leadin = leadin
        self.verbose = verbose
        self.flag_DC = False
        self.flag_df = False
        # debug outputs
        self.gamma = None
        self.x_end = None
        self.z_side1 = None
        self.z_side2 = None

        IEEEP370.__init__(self, dummies, name, *args, **kwargs)
        self.s_side1, self.s_side2 = self.split2xthru(self.s2xthru.copy(),
                                                      self.sfix_dut_fix)

    def deembed(self, ntwk: Network) -> Network:
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            FIX-DUT-FIX network from which FIX-1 and FIX-2 fixtures needs to
            be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.s2xthru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            s_side1, s_side2 = self.split2xthru(s2xthru,
                                                      self.sfix_dut_fix)
        else:
            s_side1 = self.s_side1
            s_side2 = self.s_side2

        return s_side1.inv ** ntwk ** s_side2.flipped().inv


    def makeErrorBox_v7(self, s_dut: Network, s2x: Network, gamma: ndarray,
                        z0: float, pullback:int) -> (Network, Network):
        """
        Extract the fixtures on both sides.
        """
        f = s2x.frequency.f
        n = len(f)
        s212x = s2x.s[:, 1, 0]
        DC21 = IEEEP370.dc_interp(s212x, f)
        x = np.argmax(irfft(concatenate(([DC21], s212x))))
        self.x_end = x - pullback # index of last TDR point of fixture
        #define relative length
        #python first index is 0, thus 1 should be added to get the length
        l = 1. / (2 * (x + 1))
        #define the reflections to be mimicked
        s11dut = s_dut.s[:, 0, 0]
        s22dut = s_dut.s[:, 1, 1]
        if self.verbose:
            z1 = IEEEP370.getz(s11dut, f, z0)
            z2 = IEEEP370.getz(s22dut, f, z0)
        #peel the fixture away and create the fixture model
        #python range to n-1, thus 1 to be added to have proper iteration number
        for i in range(self.x_end + 1):
            zline1 = IEEEP370.getz(s11dut, f, z0)[0]
            zline2 = IEEEP370.getz(s22dut, f, z0)[0]
            TL1 = self.makeTL(zline1,z0,gamma,l)
            TL2 = self.makeTL(zline2,z0,gamma,l)
            sTL1 = s_dut.copy()
            sTL1.s = TL1
            sTL2 = s_dut.copy()
            sTL2.s = TL2
            if i == 0:
                errorbox1 = sTL1
                errorbox2 = sTL2
            else:
                errorbox1 = errorbox1 ** sTL1
                errorbox2 = errorbox2 ** sTL2
            # equivalent to function removeTL(in,TL1,TL2,z0)
            # no need to flip sTL2 because it is symmetrical
            s_dut = sTL1.inv ** s_dut ** sTL2.inv
            #IEEE abcd implementation
            # abcd_TL1 = sTL1.a
            # abcd_TL2 = sTL2.a
            # abcd_in  = s_dut.a
            # for j in range(len(s_dut.frequency.f)):
            #     abcd_in[j, :, :] = np.linalg.lstsq(abcd_TL1[j, :, :].T,
            #                                        np.linalg.lstsq(abcd_TL1[j, :, :], abcd_in[j, :, :],
            #                                        rcond=None)[0].T, rcond=None)[0].T
            # s_dut.a = abcd_in
            s11dut = s_dut.s[:, 0, 0]
            s22dut = s_dut.s[:, 1, 1]
            # store fixture z for debug
            if(i == self.x_end):
                self.z_side1 = IEEEP370.getz(errorbox1.s[:, 0, 0], f, z0)
                self.z_side2 = IEEEP370.getz(errorbox2.s[:, 0, 0], f, z0)
        if self.verbose:
            zdut1 = IEEEP370.getz(s11dut, f, z0)
            zdut2 = IEEEP370.getz(s22dut, f, z0)
            fig, axs = subplots(1, 2, sharex = True, figsize=(2*6.4, 4.8))
            axs[0].plot(ifftshift(zdut1), label = 'DUT')
            axs[0].plot(ifftshift(self.z_side1), label = 'FIX-1')
            axs[0].plot(ifftshift(z1), color = 'k', linestyle = 'dashed', label = 'FIX-DUT-FIX')
            axs[0].set_xlim((n-50, n+x*2+50))
            axs[0].legend()
            axs[0].set_title('Left')
            axs[0].set_ylabel('Z (ohm)')
            axs[1].plot(ifftshift(zdut2), label = 'DUT')
            axs[1].plot(ifftshift(self.z_side2), label = 'FIX-2')
            axs[1].plot(ifftshift(z2), color = 'k', linestyle = 'dashed', label = 'FIX-DUT-FIX')
            axs[1].set_xlim((n-50, n+x*2+50))
            axs[1].legend()
            axs[1].set_title('Right')
            axs[1].set_ylabel('Z (ohm)')
        return errorbox1, errorbox2.flipped()

    def makeErrorBox_v8(self, s_dut: Network, s2x: Network, gamma: ndarray,
                        z0: float, pullback: int) -> Network:
        """
        Extract the fixture only on a single side.
        """
        f = s2x.frequency.f
        n = len(f)
        s212x = s2x.s[:, 1, 0]
        # extract midpoint of 2x-thru
        DC21 = IEEEP370.dc_interp(s212x, f)
        x = np.argmax(irfft(concatenate(([DC21], s212x))))
        self.x_end = x - pullback # index of last TDR point of fixture
        #define relative length
        #python first index is 0, thus 1 should be added to get the length
        l = 1. / (2 * (x + 1))
        #define the reflections to be mimicked
        s11dut = s_dut.s[:, 0, 0]
        if self.verbose:
            z1 = IEEEP370.getz(s11dut, f, z0)
        #peel the fixture away and create the fixture model
        #python range to n-1, thus 1 to be added to have proper iteration number
        for i in range(self.x_end + 1):
            zline1 = IEEEP370.getz(s11dut, f, z0)[0]
            TL1 = self.makeTL(zline1,z0,gamma,l)
            sTL1 = s_dut.copy()
            sTL1.s = TL1
            if i == 0:
                errorbox1 = sTL1
            else:
                errorbox1 = errorbox1 ** sTL1
            # equivalent to function removeTL_side1(in,TL,z0)
            s_dut = sTL1.inv ** s_dut
            s11dut = s_dut.s[:, 0, 0]
            # store fixture z for debug
            if(i == self.x_end):
                self.z_side1 = IEEEP370.getz(errorbox1.s[:, 0, 0], f, z0)
        if self.verbose:
            zdut1 = IEEEP370.getz(s11dut, f, z0)
            fig, axs = subplots(1, 1, sharex = True, figsize=(6.4, 4.8))
            axs.plot(ifftshift(zdut1), label = 'DUT')
            axs.plot(ifftshift(self.z_side1), label = 'FIX')
            axs.plot(ifftshift(z1), color = 'k', linestyle = 'dashed', label = 'FIX-DUT-FIX')
            axs.set_xlim((n-50, n+x*2+50))
            axs.legend()
            axs.set_ylabel('Z (ohm)')
        return errorbox1


    def split2xthru(self, s2xthru: Network, sfix_dut_fix: Network) -> (Network, Network):
        """
        Perform the fixtures extraction.
        """
        f = sfix_dut_fix.frequency.f
        s = sfix_dut_fix.s

        # check for bad inputs
        # check for DC point
        if(f[0] == 0):
            warnings.warn(
                "DC point detected. The included DC point will not be used during extraction.",
                RuntimeWarning, stacklevel=2
                )
            self.flag_DC = True
            f = f[1:]
            s = s[1:]
            sfix_dut_fix = Network(frequency = Frequency.from_f(f, 'Hz'), s = s)
            s2xthru.interpolate_self(Frequency.from_f(f, 'Hz'))

        # check for bad frequency vector
        df = f[1] - f[0]
        tol = 0.1 # allow a tolerance of 0.1 from delta-f to starting f (prevent non-issues from precision)
        if(np.abs(f[0] - df) > tol):
            warnings.warn(
               """Non-uniform frequency vector detected. An interpolated S-parameter matrix will be created for
               this calculation. The output results will be re-interpolated to the original vector.""",
               RuntimeWarning, stacklevel=2
               )
            self.flag_df = True
            f_original = f
            projected_n = np.floor(f[-1]/f[0])
            fnew = f[0] * (np.arange(0, projected_n) + 1)
            f_interp = Frequency.from_f(fnew, unit = 'Hz')
            sfix_dut_fix.interpolate_self(f_interp, kind = 'cubic',
                                   fill_value = 'extrapolate')
            s2xthru.interpolate_self(f_interp, kind = 'cubic',
                                   fill_value = 'extrapolate')
            f = fnew

        # check if 2x-thru is not the same frequency vector as the
        # fixture-dut-fixture
        if(not np.array_equal(sfix_dut_fix.frequency.f, s2xthru.frequency.f)):
            s2xthru.interpolate(sfix_dut_fix.frequency, kind = 'cubic',
                                   fill_value = 'extrapolate')
            warnings.warn(
               """2x-thru does not have the same frequency vector as the fixture-dut-fixture.
               Interpolating to fix problem.""",
               RuntimeWarning, stacklevel=2
               )

        # enforce Nyquist rate point
        if self.NRP_enable:
            sfix_dut_fix, TD = IEEEP370.NRP(sfix_dut_fix)
            s2xthru, _ = IEEEP370.NRP(s2xthru, -TD)

        # remove lead-in points
        if self.leadin > 0:
            _, temp1, temp2 = IEEEP370.peelNPointsLossless(
                IEEEP370.shiftNPoints(sfix_dut_fix, self.leadin), self.leadin,
                z0 = self.z0)
            leadin1 = IEEEP370.shiftOnePort(temp1, -self.leadin, 0)
            leadin2 = IEEEP370.shiftOnePort(temp2, -self.leadin, 1)
            if self.verbose:
                fig, axs = subplots(1, 2, sharex = True, figsize=(2*6.4, 4.8))
                n = len(f)
                zfdf1 = IEEEP370.getz(sfix_dut_fix.s[:, 0, 0], f, self.z0)
                zlead1 = IEEEP370.getz(leadin1.s[:, 0, 0], f, self.z0)
                axs[0].plot(ifftshift(zfdf1), label = 'FDF')
                axs[0].plot(ifftshift(zlead1), label = f'leadin ({self.leadin})')
                axs[0].set_xlim((n-self.leadin-10, n+10))
                axs[0].legend()
                axs[0].set_title('Left leadin')
                axs[0].set_ylabel('Z (ohm)')
                zfdf2 = IEEEP370.getz(sfix_dut_fix.s[:, 1, 1], f, self.z0)
                zlead2 = IEEEP370.getz(leadin2.s[:, 1, 1], f, self.z0)
                axs[1].plot(ifftshift(zfdf2), label = 'FDF')
                axs[1].plot(ifftshift(zlead2), label = f'leadin ({self.leadin})')
                axs[1].set_xlim((n-self.leadin-10, n+10))
                axs[1].legend()
                axs[1].set_title('Right leadin')
                axs[1].set_ylabel('Z (ohm)')

        # calculate gamma
        #grabbing s21
        s212x = s2xthru.s[:, 1, 0]
        #get the attenuation and phase constant per length
        beta_per_length = -unwrap(angle(s212x))
        # because lossless would be abs(S11)**2 + abs(S21)**2 = 1
        attenuation = np.abs(s2xthru.s[:,1,0])**2 / (1. - np.abs(s2xthru.s[:,0,0])**2)
        alpha_per_length = (10.0 * np.log10(attenuation)) / -8.686 # not 20 * log10() because of **2 above
        if self.bandwidth_limit == 0:
            #divide by 2*n + 1 to get prop constant per discrete unit length
            self.gamma = alpha_per_length + 1j * beta_per_length # gamma without DC
        else:
            #fit the attenuation up to the limited bandwidth
            bwl_x = np.argmin(np.abs(f - self.bandwidth_limit))
            X = np.array([np.sqrt(f[0:bwl_x+1]), f[0:bwl_x+1], f[0:bwl_x+1]**2])
            b = np.linalg.lstsq(X.conj().T, alpha_per_length[0:bwl_x+1], rcond=None)[0]
            alpha_per_length_fit = b[0] * np.sqrt(f) + b[1] * f + b[2] * f**2
            #divide by 2*n + 1 to get prop constant per discrete unit length
            self.gamma = alpha_per_length_fit + 1j * beta_per_length # gamma without DC
        if self.verbose:
            fig, axs = subplots(1, 2, figsize=(2*6.4, 4.8))
            fig.suptitle('Gamma determination')
            axs[0].plot(s2xthru.frequency.f_scaled, alpha_per_length, label = 'alpha per length')
            if self.bandwidth_limit != 0:
                f_bw_hz = self.bandwidth_limit
                f_bw = f_bw_hz / s2xthru.frequency.multiplier
                unit = s2xthru.frequency.unit
                alpha_bw = b[0] * np.sqrt(f_bw_hz) + b[1] * f_bw_hz + b[2] * f_bw_hz**2
                axs[0].plot(s2xthru.frequency.f_scaled, alpha_per_length_fit,
                            label = 'alpha per length fit')
                axs[0].plot([f_bw], [alpha_bw], color = 'k', marker = 'o', linestyle = None,
                            label = f'bandwidth_limit = {f_bw} {unit}')
            axs[0].legend()
            axs[0].set_xlabel(f'Frequency ({s2xthru.frequency.unit})')
            axs[0].set_ylabel('Alpha (Neper/length)')
            axs[1].plot(s2xthru.frequency.f_scaled, beta_per_length, label = 'beta per length')
            axs[1].set_xlabel(f'Frequency ({s2xthru.frequency.unit})')
            axs[1].set_ylabel('Beta (rad/length)')
            axs[1].legend()

        # extract error boxes
        # make the both error box
        s_side1 = IEEEP370.thru(sfix_dut_fix)
        s_side2 = IEEEP370.thru(sfix_dut_fix)

        # In the implementation, FIX-2 is flipped.
        # This does not met IEEEP370 numbering recommendation but is left as
        # is for comparison ease.
        if self.pullback1 == self.pullback2 and self.side1 and self.side2:
            (s_side1, s_side2) = self.makeErrorBox_v7(sfix_dut_fix, s2xthru,
                                  self.gamma, self.z0, self.pullback1)
        elif self.side1 and self.side2:
            s_side1 = self.makeErrorBox_v8(sfix_dut_fix, s2xthru,
                                   self.gamma, self.z0, self.pullback1)
            s_side2 = self.makeErrorBox_v8(sfix_dut_fix.flipped(),s2xthru,
                                   self.gamma, self.z0, self.pullback2)
            s_side2 = s_side2.flipped()
        elif self.side1:
            s_side1 = self.makeErrorBox_v8(sfix_dut_fix, s2xthru,
                                   self.gamma, self.z0, self.pullback1)
        elif self.side2:
            s_side2 = self.makeErrorBox_v8(sfix_dut_fix.flipped(),s2xthru,
                                   self.gamma, self.z0, self.pullback2)
            s_side2 = s_side2.flipped()
        else:
            warnings.warn(
               "no output because no output was requested",
               RuntimeWarning, stacklevel=2
               )


        # interpolate to original frequency if needed
        # revert back to original frequency vector
        if self.flag_df:
            f_interp = Frequency.from_f(f_original, unit = 'Hz')
            s_side1.interpolate_self(f_interp, kind = 'cubic',
                                   fill_value = 'extrapolate')
            s_side2.interpolate_self(f_interp, kind = 'cubic',
                                   fill_value = 'extrapolate')

        # add DC back in
        if self.flag_DC:
            s_side1 = IEEEP370.add_dc(s_side1)
            s_side2 = IEEEP370.add_dc(s_side2)

        # remove lead in
        if self.leadin > 0:
            s_side1 = leadin1 ** s_side1
            s_side2 = s_side2 ** leadin2

        # if Nyquist Rate Point enforcement is enabled
        if self.NRP_enable:
            s_side1, _ = IEEEP370.NRP(s_side1, TD, 0)
            s_side2, _ = IEEEP370.NRP(s_side2, TD, 1)

        # unflip FIX-2 as per IEEEP370 numbering recommendation
        return (s_side1, s_side2.flipped())

    def plot_check_residuals(self, ax: Axes = None) -> (Figure, Axes):
        res = self.deembed(self.s2xthru)
        res.name = 'Residuals'

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #1: Self de-embedding of 2X-Thru')

        ax[0].set_title('Magnitude residuals')
        res.plot_s_db(1,0, ax = ax[0], color = '0.5')
        res.plot_s_db(0,1, ax = ax[0], color = 'k')
        ax[0].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [0.1, 0.1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[0].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [-0.1, -0.1],
                       linestyle = 'dashed', color = 'r')
        ax[0].legend(loc = 'upper right')

        ax[1].set_title('Phase residuals')
        res.plot_s_deg(1,0, ax = ax[1], color = '0.5')
        res.plot_s_deg(0,1, ax = ax[1], color = 'k')
        ax[1].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [1, 1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[1].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [-1, -1],
                       linestyle = 'dashed', color = 'r')
        ax[1].legend(loc = 'upper right')
        fig.tight_layout()

        return (fig, ax)

    def plot_check_impedance(self, fix_dut_fix: Network = None, ax: Axes = None,
                             window: str = 'hamming') -> (Figure, Axes):
        # if dc point already exists, it will be replaced
        s2xthru = IEEEP370.extrapolate_to_dc(self.s2xthru)
        fix1 = IEEEP370.extrapolate_to_dc(self.s_side1)
        fix2 = IEEEP370.extrapolate_to_dc(self.s_side2)
        if fix_dut_fix is not None:
            fix_dut_fix = IEEEP370.extrapolate_to_dc(fix_dut_fix)
        else:
            fix_dut_fix = IEEEP370.extrapolate_to_dc(self.sfix_dut_fix)
        n = s2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * s2xthru.frequency.step) # ns

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #2: Compare the TDR of the fixture model to the FIX-DUT-FIX')
        ax[0].set_title('Side 1')
        fix1.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], color = 'k')
        y = ax[0].lines[-1].get_ydata()
        s2xthru.plot_z_time_step(0, 0, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], linestyle = 'dashed', color = 'm')
        ax[0].plot([-self.leadin * dt], [y[n // 2 - self.leadin]], marker = 's', color = 'k',
                   label = f'start (leadin = {self.leadin})')
        ax[0].plot([self.x_end * dt], [y[self.x_end + n // 2]], marker = 'o', color = 'k',
                   label = f'end (pullback1 = {self.pullback1})')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax[0].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[0].lines[2].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ymin = np.min(np.array([ax[0].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[0].lines[2].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ax[0].set_ylim((ymin - 5, ymax + 5))
        ax[0].legend(loc = 'lower left')

        ax[1].set_title('Side 2')
        fix2.plot_z_time_step(0, 0, window = window,
                              ax = ax[1], color = 'k')
        y = ax[1].lines[-1].get_ydata()
        s2xthru.plot_z_time_step(1, 1, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(1, 1, window = window,
                              ax = ax[1], linestyle = 'dashed', color = 'm')
        ax[1].plot([-self.leadin * dt], [y[n // 2 - self.leadin]], marker = 's', color = 'k',
                  label = f'start (leadin = {self.leadin})')
        ax[1].plot([self.x_end * dt], [y[self.x_end + n // 2]], marker = 'o', color = 'k',
                   label = f'end (pullback2 = {self.pullback2})')
        # fit the plot around fix and 2x-thru in case FIX-DUT-FIX is much larger
        ymax = np.max(np.array([ax[1].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[1].lines[2].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ymin = np.min(np.array([ax[1].lines[0].get_ydata()[(n // 2):(self.x_end + n // 2)],
                               ax[1].lines[2].get_ydata()[(n // 2):(self.x_end + n // 2)]]))
        ax[1].set_ylim((ymin - 5, ymax + 5))
        delay = 2 * self.x_end * dt
        ax[1].set_xlim((-0.5 * delay, 1.5 * delay))
        ax[1].legend(loc = 'lower left')

        fig.tight_layout()

        return (fig, ax)


class IEEEP370_MM_ZC_2xThru(IEEEP370):
    """
    Creates error boxes from a 4-port from 2x-Thru and FIX-DUT-FIX networks.

    Based on [I3E370]_.

    A deembedding object is created with 2x-Thru (FIX-FIX) and FIX-DUT-FIX
    measurements, which are split into left (FIX-1_2) and right (FIX-3_4)
    fixtures with IEEEP370 Zc2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1_2
    and FIX-3_4 are deembedded from FIX_DUT_FIX network.

    This method is applicable only when there is 2xThru and FIX_DUT_FIX
    networks.

    The possible difference of impedance between 2x-Thru and FIX-DUT-FIX
    is corrected.

    The algorithm computes the length of the fixtures by halving the delay of
    2x-Thru in time domain transmission. The propagation constant gamma is also
    determined from the 2xThru. It then peels the FIX-DUT-FIX time domain
    impedance profile iteratively in cycles of determining start impedance and
    deembedding a single time sample long transmission line.

    Note
    ----
    The `port_order` ='first', means front-to-back also known as odd/even,
    while `port_order`='second' means left-to-right also known as sequential.
    `port_order`='third' means yet another numbering method.
    Next figure show example of port numbering with 4-port networks.

    The `scikit-rf` cascade ** 2N-port operator use second scheme. This is very
    convenient to write compact deembedding and other expressions.

    numbering diagram::

      port_order = 'first'
           +---------+
          -|0       1|-
          -|2       3|-
           +---------+

      port_order = 'second'
           +---------+
          -|0       2|-
          -|1       3|-
           +---------+

      port_order = 'third'
           +---------+
          -|0       3|-
          -|1       2|-
           +---------+

    use `Network.renumber` to change port ordering.

    Example
    --------
    >>> import skrf as rf
    >>> from skrf.calibration import IEEEP370_MM_ZC_2xThru

    Create network objects for 2xThru and FIX-DUT-FIX

    >>> s2xthru = rf.Network('2xthru.s4p')
    >>> fdf = rf.Network('f-dut-f.s4p')

    Create de-embedding object

    >>> dm = IEEEP370_MM_ZC_2xThru(dummy_2xthru = s2xthru,
                             dummy_fix_dut_fix = fdf,
                             bandwidth_limit = 10e9,
                             pullback1 = 0, pullback2 = 0,
                             leadin = 0,
                             name = 'zc2xthru')

    Apply deembedding to get the DUT

    >>> dut = dm.deembed(fdf)

    Note
    ----
    numbering diagram::

         FIX-1_2  DUT      FIX-3_4
         +----+   +----+   +----+
        -|1  3|---|1  3|---|3  1|-
        -|2  4|---|2  4|---|4  2|-
         +----+   +----+   +----+


    Warning
    -------
    There is one difference compared to the original matlab implementation
    [I3E370]:
        - FIX-2 is flipped (see diagram above)

    References
    ----------
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370mmZc2xthru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6

    """
    def __init__(self, dummy_2xthru: Network, dummy_fix_dut_fix: Network,
                 name: str = None,
                 z0: float = 50, port_order: PortOrderT = 'second',
                 bandwidth_limit: float = 0,
                 pullback1: int = 0, pullback2: int = 0,
                 side1: bool = True, side2: bool = True,
                 NRP_enable: bool = True, leadin: int = 1,
                 verbose: bool = False,
                 *args, **kwargs) -> None:
        """
        IEEEP370_MM_ZC_2xThru De-embedding Initializer

        Parameters
        -----------

        dummy_2xthru : :class:`~skrf.network.Network` object
            2xThru (FIX-FIX) network.

        name : string
            optional name of de-embedding object

        z0 :
            reference impedance of the S-parameters (default: 50)

        port_order : ['first', 'second', 'third']
            specify what numbering scheme to use. See above. (default: second)

        bandwidth_limit :
            max frequency for a fitting function
            (default: 0, use all s-parameters without fit)

        pullback1, pullback2 :
            a number of discrete points to leave in the fixture on side 1
            respectively on side 2 (default: 0 leave all)

        side1, side2 :
            set to de-embed the side1 resp. side2 errorbox (default: True)

        NRP_enable :
            set to enforce the Nyquist Rate Point during de-embedding and to
            add the appropriote delay to the errorboxes (default: True)

        leadin :
            a number of discrete points before t = 0 that are non-zero from
            calibration error (default: 1)

        verbose :
            view the process (default: False)

        args, kwargs:
            passed to :func:`Deembedding.__init__`

        See Also
        ---------
        :func:`Deembedding.__init__`

        """
        self.s2xthru = dummy_2xthru.copy()
        self.sfix_dut_fix = dummy_fix_dut_fix.copy()
        dummies = [self.s2xthru]
        self.z0 = z0
        self.port_order = port_order
        self.bandwidth_limit = bandwidth_limit
        self.pullback1 = pullback1
        self.pullback2 = pullback2
        self.side1 = side1
        self.side2 = side2
        self.NRP_enable = NRP_enable
        self.leadin = leadin
        self.verbose = verbose
        self.flag_DC = False
        self.flag_df = False

        # debug outputs
        self.gamma_dd = None
        self.x_end_dd = None
        self.z_side1_dd = None
        self.z_side2_dd = None
        self.gamma_cc = None
        self.x_end_cc = None
        self.z_side1_cc = None
        self.z_side2_cc = None

        IEEEP370.__init__(self, dummies, name, *args, **kwargs)
        self.se_side1, self.se_side2 = self.split2xthru(self.s2xthru,
                                                        self.sfix_dut_fix)

    def deembed(self, ntwk: Network) -> Network:
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            FIX-DUT-FIX network from which FIX-1_2 and FIX-3_4 fixtures needs
            to be removed via de-embedding

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding

        """

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.s2xthru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            se_side1, se_side2 = self.split2xthru(s2xthru, self.sfix_dut_fix)
        else:
            se_side1 = self.se_side1
            se_side2 = self.se_side2

        # check if 4-port
        if ntwk.nports != 4:
            raise(ValueError('2xthru has to be a 4-port network.'))
        # renumber if required
        if self.port_order == 'first':
            N = ntwk.nports
            old_order = list(range(N))
            new_order = list(range(0, N, 2)) + list(range(1, N, 2))
            ntwk.renumber(old_order, new_order)
        elif self.port_order == 'third':
            N = ntwk.nports
            old_order = list(range(N))
            new_order = list(range(0, N//2)) + list(range(N-1, N//2-1, -1))
            ntwk.renumber(old_order, new_order)

        deembedded = se_side1.inv ** ntwk ** se_side2.flipped().inv
        #renumber back if required
        if self.port_order != 'second':
            deembedded.renumber(new_order, old_order)
        return deembedded

    def split2xthru(self, se_2xthru: Network, se_fdf: Network) -> (Network, Network):
        """
        Perform the fixtures extraction.
        """
        # check if 4-port
        if se_2xthru.nports != 4 or se_fdf.nports != 4:
            raise(ValueError('2xthru has to be a 4-port network.'))
        # renumber if required
        if self.port_order == 'first':
            N = se_2xthru.nports
            old_order = list(range(N))
            new_order = list(range(0, N, 2)) + list(range(1, N, 2))
            se_2xthru.renumber(old_order, new_order)
            se_fdf.renumber(old_order, new_order)
        elif self.port_order == 'third':
            N = se_2xthru.nports
            old_order = list(range(N))
            new_order = list(range(0, N//2)) + list(range(N-1, N//2-1, -1))
            se_2xthru.renumber(old_order, new_order)
            se_fdf.renumber(old_order, new_order)

        #convert to mixed-modes
        mm_2xthru = se_2xthru.copy()
        mm_2xthru.se2gmm(p = 2)
        mm_fdf = se_fdf.copy()
        mm_fdf.se2gmm(p = 2)

        #extract common and differential mode and model fixtures for each
        sdd = subnetwork(mm_2xthru, [0, 1])
        scc = subnetwork(mm_2xthru, [2, 3])
        sdd_fdf = subnetwork(mm_fdf, [0, 1])
        scc_fdf = subnetwork(mm_fdf, [2, 3])
        dm_dd  = IEEEP370_SE_ZC_2xThru(dummy_2xthru = sdd,
                                  dummy_fix_dut_fix = sdd_fdf,
                                  z0 = self.z0 * 2,
                                  bandwidth_limit = self.bandwidth_limit,
                                  pullback1 = self.pullback1,
                                  pullback2 = self.pullback2,
                                  side1 = self.side1,
                                  side2 = self.side2,
                                  NRP_enable = self.NRP_enable,
                                  leadin = self.leadin,
                                  verbose = self.verbose)
        # debug outputs
        self.gamma_dd = dm_dd.gamma
        self.x_end_dd = dm_dd.x_end
        self.z_side1_dd = dm_dd.z_side1
        self.z_side2_dd = dm_dd.z_side2
        dm_cc  = IEEEP370_SE_ZC_2xThru(dummy_2xthru = scc,
                                  dummy_fix_dut_fix = scc_fdf,
                                  z0 = self.z0 / 2,
                                  bandwidth_limit = self.bandwidth_limit,
                                  pullback1 = self.pullback1,
                                  pullback2 = self.pullback2,
                                  side1 = self.side1,
                                  side2 = self.side2,
                                  NRP_enable = self.NRP_enable,
                                  leadin = self.leadin,
                                  verbose = self.verbose)
        # debug outputs
        self.gamma_cc = dm_cc.gamma
        self.x_end_cc = dm_cc.x_end
        self.z_side1_cc = dm_cc.z_side1
        self.z_side2_cc = dm_cc.z_side2
        #convert back to single-ended
        mm_side1 = concat_ports([dm_dd.s_side1, dm_cc.s_side1], port_order = 'first')
        se_side1 = mm_side1.copy()
        se_side1.gmm2se(p = 2)
        mm_side2 = concat_ports([dm_dd.s_side2, dm_cc.s_side2], port_order = 'first')
        se_side2 = mm_side2.copy()
        se_side2.gmm2se(p = 2)

        return (se_side1, se_side2)

    def plot_check_residuals(self, ax: Axes = None) -> (Figure, Axes):
        res = self.deembed(self.s2xthru)
        res.name = 'Residuals'
        res.se2gmm(p=2)

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #1: Self de-embedding of 2X-Thru')

        ax[0].set_title('Magnitude residuals')
        res.plot_s_db(1,0, ax = ax[0])
        res.plot_s_db(0,1, ax = ax[0])
        res.plot_s_db(3,2, ax = ax[0])
        res.plot_s_db(2,3, ax = ax[0])
        ax[0].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [0.1, 0.1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[0].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [-0.1, -0.1],
                       linestyle = 'dashed', color = 'r')
        ax[0].legend(loc = 'upper right')

        ax[1].set_title('Phase residuals')
        res.plot_s_deg(1,0, ax = ax[1])
        res.plot_s_deg(0,1, ax = ax[1])
        res.plot_s_deg(3,2, ax = ax[1])
        res.plot_s_deg(2,3, ax = ax[1])
        ax[1].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [1, 1],
                       linestyle = 'dashed', color = 'r', label = 'Limit')
        ax[1].plot([res.frequency.f[0], res.frequency.f[-1]],
                       [-1, -1],
                       linestyle = 'dashed', color = 'r')
        ax[1].legend(loc = 'upper right')
        fig.tight_layout()

        return (fig, ax)

    def plot_check_impedance(self, fix_dut_fix: Network = None, ax: Axes = None,
                             window: str = 'hamming') -> (Figure, Axes):
        # if dc point already exists, it will be replaced
        s2xthru = self.s2xthru.copy()
        s2xthru.se2gmm(p=2)
        s2xthru = IEEEP370.extrapolate_to_dc(s2xthru)
        fix1 = self.se_side1.copy()
        fix1.se2gmm(p=2)
        fix1 = IEEEP370.extrapolate_to_dc(fix1)
        fix2 = self.se_side2.copy()
        fix2.se2gmm(p=2)
        fix2 = IEEEP370.extrapolate_to_dc(fix2)
        if fix_dut_fix is not None:
            fix_dut_fix = fix_dut_fix.copy()
            fix_dut_fix.se2gmm(p=2)
            fix_dut_fix = IEEEP370.extrapolate_to_dc(fix_dut_fix)
        else:
            fix_dut_fix = self.sfix_dut_fix.copy()
            fix_dut_fix.se2gmm(p=2)
            fix_dut_fix = IEEEP370.extrapolate_to_dc(fix_dut_fix)
        n = s2xthru.frequency.npoints * 2 - 1
        dt = 1e9 / (n * s2xthru.frequency.step) # ns

        if ax is None:
            fig, ax = subplots(1, 2, sharex = True, figsize=(10, 5))
        else:
            fig = ax.get_figure()

        fig.suptitle('Consistency test #2: Compare the TDR of the fixture model to the FIX-DUT-FIX')
        ax[0].set_title('Side 1')
        fix1.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], color = 'k')
        y = ax[0].lines[-1].get_ydata()
        s2xthru.plot_z_time_step(0, 0, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(0, 0, window = window,
                              ax = ax[0], linestyle = 'dashed', color = 'm')
        ax[0].plot([-self.leadin * dt], [y[n // 2 - self.leadin]], marker = 's', color = 'k',
                   label = f'start (leadin = {self.leadin})')
        ax[0].plot([self.x_end_dd * dt], [y[self.x_end_dd + n // 2]], marker = 'o', color = 'k',
                   label = f'end (pullback1 = {self.pullback1})')
        fix1.plot_z_time_step(2, 2, window = window,
                              ax = ax[0], color = 'k')
        s2xthru.plot_z_time_step(2, 2, window = window,
                                 ax = ax[0], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(2, 2, window = window,
                                     ax = ax[0], linestyle = 'dashed', color = 'b')
        ax[0].legend(loc = 'center left')

        ax[1].set_title('Side 2')
        fix2.plot_z_time_step(0, 0, window = window,
                              ax = ax[1], color = 'k')
        y = ax[1].lines[-1].get_ydata()
        s2xthru.plot_z_time_step(1, 1, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(1, 1, window = window,
                              ax = ax[1], linestyle = 'dashed', color = 'm')
        ax[1].plot([-self.leadin * dt], [y[n // 2 - self.leadin]], marker = 's', color = 'k',
                  label = f'start (leadin = {self.leadin})')
        ax[1].plot([self.x_end_dd * dt], [y[self.x_end_dd + n // 2]], marker = 'o', color = 'k',
                   label = f'end (pullback2 = {self.pullback2})')
        fix2.plot_z_time_step(2, 2, window = window,
                              ax = ax[1], color = 'k')
        s2xthru.plot_z_time_step(3, 3, window = window,
                                 ax = ax[1], linestyle = 'dotted', color = '0.2')
        fix_dut_fix.plot_z_time_step(3, 3, window = window,
                                     ax = ax[1], linestyle = 'dashed', color = 'b')
        delay = 2 * self.x_end_dd * dt
        ax[1].set_xlim((-0.5 * delay, 1.5 * delay))
        ax[1].legend(loc = 'center left')

        fig.tight_layout()

        return (fig, ax)


# ---------------------------------------------------------------------------
# Modal 2x-thru de-embedding of N coupled lines (extension outside IEEE 370)
# ---------------------------------------------------------------------------

# Modal transform of a differential pair, columns are (differential, common) modes:
# V1 = Vd / 2 + Vc, V2 = -Vd / 2 + Vc. It gives the same S-parameters as `Network.se2gmm(p=2)`.
_MIXED_MODE_TV = np.array([[0.5, 1.0], [-0.5, 1.0]])


def _modal_xi(tv: ndarray, z0: float, z_modal: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    r"""
    Matrices of the S-parameter change of basis between terminal and modal waves.

    The terminal voltages and currents of N lines are written as
    :math:`V = T_v V_m` and :math:`I = T_i I_m` with :math:`T_i = T_v^{-T}`, which
    conserves power and preserves reciprocity. Modal waves are defined with the
    real modal reference impedances `z_modal`. The returned matrices follow the
    convention of :meth:`skrf.network.Network.se2gmm`, that is
    :math:`S_m = (X_{21} + X_{22} S)(X_{11} + X_{12} S)^{-1}`.

    Parameters
    ----------
    tv : :class:`numpy.ndarray`, shape (F, N, N)
        Modal transform, one mode per column.
    z0 : float
        Real terminal reference impedance of the 2N ports.
    z_modal : :class:`numpy.ndarray`, shape (N,)
        Real modal reference impedance of every mode.

    Returns
    -------
    X11, X12, X21, X22 : :class:`numpy.ndarray`, shape (F, 2N, 2N)
        The same N x N block acts on the ports of the left side and of the right side.
    """
    tv = np.asarray(tv, dtype=complex)
    z_modal = np.asarray(z_modal, dtype=float)
    n_f, n, _ = tv.shape
    sqrt_z0 = np.sqrt(z0)
    u = np.linalg.inv(tv) * sqrt_z0
    w = z_modal[None, :, None] * np.swapaxes(tv, 1, 2) / sqrt_z0
    dm = 1.0 / (2.0 * np.sqrt(z_modal))[None, :, None]
    p = dm * (u + w)
    m = dm * (u - w)
    x_p = np.zeros((n_f, 2 * n, 2 * n), dtype=complex)
    x_m = np.zeros_like(x_p)
    x_p[:, :n, :n] = p
    x_p[:, n:, n:] = p
    x_m[:, :n, :n] = m
    x_m[:, n:, n:] = m
    return x_p, x_m, x_m, x_p


def _se2modal(s: ndarray, xi: tuple[ndarray, ndarray, ndarray, ndarray]) -> ndarray:
    """
    Terminal S-parameters to modal S-parameters, see :func:`_modal_xi`.
    """
    x11, x12, x21, x22 = xi
    a = x21 + x22 @ s
    b = x11 + x12 @ s
    return mf.rsolve(b, a)


def _modal2se(s_m: ndarray, xi: tuple[ndarray, ndarray, ndarray, ndarray]) -> ndarray:
    """
    Modal S-parameters to terminal S-parameters, see :func:`_modal_xi`.
    """
    x11, x12, x21, x22 = xi
    return np.linalg.solve(x22 - s_m @ x12, s_m @ x11 - x21)


def _abcd_2nport(s: ndarray, z0: float) -> ndarray:
    """
    ABCD matrix of a 2N-port in 'second' port order, [V1; I1] = ABCD [V2; -I2].

    :func:`skrf.network.s2a` handles two-ports only. Here the scattering transfer
    matrix of :func:`skrf.network.s2t`, [b1; a1] = T [a2; b2], is converted with
    a real, scalar reference impedance `z0`.

    Parameters
    ----------
    s : :class:`numpy.ndarray`, shape (F, 2N, 2N)
    z0 : float
        Real reference impedance of all the ports.

    Returns
    -------
    abcd : :class:`numpy.ndarray`, shape (F, 2N, 2N)
        N x N blocks [[A, B], [C, D]].
    """
    n = s.shape[1] // 2
    eye = np.eye(n)
    sqrt_z0 = np.sqrt(z0)
    k = np.block([[sqrt_z0 * eye, sqrt_z0 * eye], [-eye / sqrt_z0, eye / sqrt_z0]])
    return k @ s2t(s) @ np.linalg.inv(k)


def _eig_gap(lam: ndarray) -> ndarray:
    """
    Relative spacing of the eigenvalues, shape (F,).

    For each pair of eigenvalues, ``|l_i - l_j| / max(|l_i|, |l_j|)``; the result
    is the smallest over all the pairs. It is ``inf`` when there is a single mode.
    """
    n_f, n = lam.shape
    gap = np.full(n_f, np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            scale = np.maximum(np.maximum(np.abs(lam[:, i]), np.abs(lam[:, j])), 1e-30)
            gap = np.minimum(gap, np.abs(lam[:, i] - lam[:, j]) / scale)
    return gap


def _track_modes(vec: ndarray, lam: ndarray, start: int) -> tuple[ndarray, ndarray]:
    """
    Make eigenvectors continuous over frequency (in place).

    Starting at index `start`, where the modes are the best separated, and going
    outward in both directions, the columns of every frequency are matched to the
    previous one with a linear assignment on the overlap magnitudes, then their sign
    is chosen so that the overlap is positive. The loop is sequential by nature.
    """
    n_f = vec.shape[0]
    for indices in (range(start + 1, n_f), range(start - 1, -1, -1)):
        prev = start
        for i in indices:
            cost = -np.abs(vec[prev].conj().T @ vec[i])
            _, col = linear_sum_assignment(cost)
            vec[i] = vec[i][:, col]
            lam[i] = lam[i][col]
            sign = np.sign(np.real(np.einsum('ik,ik->k', vec[prev].conj(), vec[i])))
            sign[sign == 0] = 1.0
            vec[i] = vec[i] * sign
            prev = i
    return vec, lam


def _transform_ripple(vec: ndarray, f: ndarray, deg: int = 3) -> float:
    """
    Largest deviation of the modal vectors from a low-order polynomial of frequency.

    The real and imaginary parts of every entry of `vec` (shape (F, N, N)) are fitted
    with a polynomial of degree `deg` in `f`. The result is about 1e-8 for a uniform
    coupled line and several 1e-2 or more for fixtures that are not line-like.
    """
    if len(f) < 2 * (deg + 2):
        return 0.0
    x = f / f[-1]
    data = vec.reshape(len(f), -1)
    data = np.concatenate([data.real, data.imag], axis=1)
    vander = np.vander(x, deg + 1)
    coef = np.linalg.lstsq(vander, data, rcond=None)[0]
    return float(np.max(np.abs(data - vander @ coef)))


def _eig_modal_transform(s2x: Network, z0: float, mode: str = 'const',
                         eig_scaling: str = 'unit') -> tuple[ndarray, dict]:
    r"""
    Estimate the modal transform of a mirror-symmetric 2N-port 2x-thru.

    For a reciprocal fixture-fixture network that is the mirror image of
    itself, the A block of its ABCD matrix is :math:`A = T_v \cosh(\gamma \ell) T_v^{-1}`
    when it behaves as N uniform coupled lines, so the columns of :math:`T_v` are
    the eigenvectors of A.

    Parameters
    ----------
    s2x : :class:`~skrf.network.Network`
        2x-thru, 2N-port, 'second' port order.
    z0 : float
        Real reference impedance of the 2x-thru.
    mode : {'const', 'freq'}
        'freq' returns the eigenvectors of every frequency point, after tracking them
        over frequency. 'const' returns a single transform, the average of the real
        part of the eigenvectors weighted by the relative eigenvalue spacing, which
        is much less sensitive to noise.
    eig_scaling : {'unit', 'image'}
        Column scaling of the 'freq' transform. 'unit': unit norm, with
        :math:`v^T v` real and positive. 'image': scaled so that the modal image
        impedance is constant.

    Returns
    -------
    tv : :class:`numpy.ndarray`, shape (F, N, N)
        Modal transform, one mode per column.
    info : dict
        'eigenvalues' (F, N), 'eig_gap' (F,), 'mode_order' (N,), 'ripple' (float).
        A DC point, if any, is excluded from the estimation and takes the values of the
        first other frequency.
    """
    if mode not in ('const', 'freq'):
        raise ValueError(f"mode must be 'const' or 'freq', not {mode!r}.")
    if eig_scaling not in ('unit', 'image'):
        raise ValueError(f"eig_scaling must be 'unit' or 'image', not {eig_scaling!r}.")
    f = s2x.f
    s = s2x.s
    n = s.shape[1] // 2
    valid = np.flatnonzero(f > 0)
    if len(valid) == 0:
        raise ValueError('The 2x-thru has no frequency point above DC.')

    abcd = _abcd_2nport(s[valid], z0)
    lam, vec = np.linalg.eig(abcd[:, :n, :n])
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec * np.exp(-0.5j * np.angle(np.einsum('fik,fik->fk', vec, vec)))[:, None, :]
    gap = _eig_gap(lam)
    vec, lam = _track_modes(vec, lam, int(np.argmax(gap)))

    # deterministic order: the slowest mode (smallest Re(cosh) at the lowest frequency) first
    order = np.argsort(lam[0].real, kind='stable')
    vec = vec[:, :, order]
    lam = lam[:, order]
    ripple = _transform_ripple(vec, f[valid])

    if mode == 'const':
        weight = np.where(np.isfinite(gap), gap, 1.0)
        if not weight.sum() > 0:
            weight = np.ones_like(weight)
        const = np.einsum('f,fik->ik', weight / weight.sum(), vec.real)
        const = const / np.linalg.norm(const, axis=0, keepdims=True)
        vec = np.broadcast_to(const, vec.shape).astype(complex)
    elif eig_scaling == 'image':
        ti = np.linalg.inv(np.swapaxes(vec, 1, 2))
        b_m = np.linalg.inv(vec) @ abcd[:, :n, n:] @ ti
        c_m = np.linalg.inv(ti) @ abcd[:, n:, :n] @ vec
        z_k = np.sqrt(np.diagonal(b_m, axis1=1, axis2=2) / np.diagonal(c_m, axis1=1, axis2=2))
        z_k = np.where(z_k.real < 0, -z_k, z_k)
        z_ref = np.median(np.abs(z_k), axis=0)
        vec = vec * np.sqrt(z_k / z_ref)[:, None, :]

    tv = np.empty((len(f), n, n), dtype=complex)
    tv[:] = vec[0]
    tv[valid] = vec
    eigenvalues = np.empty((len(f), n), dtype=complex)
    eigenvalues[:] = lam[0]
    eigenvalues[valid] = lam
    eig_gap = np.full(len(f), gap[0])
    eig_gap[valid] = gap
    info = {'eigenvalues': eigenvalues, 'eig_gap': eig_gap, 'mode_order': order, 'ripple': ripple}
    return tv, info


def _modal_references(tv: ndarray, z0: float) -> ndarray:
    """
    Real modal reference impedances that go with the modal transform `tv`.

    For the mode k, ``z0 * median_f(norm(Ti[:, k]) / norm(Tv[:, k]))`` with ``Ti = Tv^-T``.
    For the mixed-mode transform of a differential pair it gives exactly ``2 * z0``
    (differential) and ``z0 / 2`` (common).
    """
    ti = np.linalg.inv(np.swapaxes(tv, 1, 2))
    ratio = np.linalg.norm(ti, axis=1) / np.linalg.norm(tv, axis=1)
    return z0 * np.median(ratio, axis=0)


class Modal_NZC_2xThru(IEEEP370):
    r"""
    Creates error boxes from a 2N-port test fixture 2x-thru, for N coupled lines.

    Based on [ElSA20]_ and [I3E370]_, but **not part of the IEEE 370 standard**.

    A deembedding object is created with a single 2x-thru (FIX-FIX) network of
    N coupled lines, that is 2N ports. The 2x-thru is transformed to modal
    S-parameters, each mode is split into a left (FIX-1) and a right (FIX-2) half
    with the single-ended IEEEP370 NZC method (:class:`IEEEP370_SE_NZC_2xThru`),
    and the two halves are transformed back to terminal S-parameters.

    When :func:`Deembedding.deembed` is applied, the S-parameters of FIX-1 and
    FIX-2 are deembedded from the FIX-DUT-FIX network.

    For a differential pair (N = 2) with ``modal_transform='mixed'``, this is the
    method of :class:`IEEEP370_MM_NZC_2xThru`: the modal transform is then the
    differential and common mode transform. The other options are meant for
    N > 2 lines, or for pairs whose modes are not the differential and
    common modes.

    Warning
    -------
    This is an extension that goes beyond IEEE 370. The standard uses only the
    differential-differential and common-common blocks (Sdd and Scc) of a
    differential 2x-thru, and its fixture electrical requirement FER6 limits the
    mode conversion of the fixture. Here the full modal 2N-port is split.

    The single-line splitting of each mode is valid when the 2x-thru behaves as
    N uniform coupled lines in the modal basis, so that each mode propagates as
    its own transmission line. Mode conversion generated by the fixture
    (launches that are not mirror-symmetric between the lines, asymmetric
    discontinuities, and so on) breaks this assumption.
    The following attributes are set to help judge it, and a
    :class:`RuntimeWarning` starting with ``Modal 2x-thru:`` is issued when a
    limit is exceeded:

    - ``mode_conversion_db``: worst cross-mode transmission minus the insertion
      loss of the mode, per frequency, in the modal basis (like FER6 for a
      differential pair). The limit is `conversion_limit_db`.
    - ``transform_ripple`` (limit ``transform_ripple_limit``, warns for
      ``'eig'`` only): deviation of the per-frequency modal vectors from a smooth
      polynomial of frequency.
    - ``mirror_asymmetry`` (limit ``mirror_asymmetry_limit``): how far the
      2x-thru is from being the mirror image of itself.
    - ``eig_gap`` (limit ``eig_gap_limit``, eigenvector options only): relative
      spacing of the modal eigenvalues.

    The limits are class attributes and can be changed. They were set on a few
    synthetic fixtures and are not a certification of the result.

    De-embedding the 2x-thru itself to a thru is **not** a quality check for
    ``modal_transform='eig'``. The eigenvectors of a reciprocal, mirror-symmetric
    2x-thru always decouple it exactly, whether or not the fixture is
    line-like. Compare a known device, or the diagnostics above, instead.

    Note
    ----
    The modal transform relates terminal and modal voltages :math:`V = T_v V_m`,
    with the modes in the columns of :math:`T_v`, and currents
    :math:`I = T_v^{-T} I_m`. It is real and constant for a uniform coupled
    line in the quasi-TEM approximation, but it is not unique: a scale of each
    column changes the modal characteristic impedance. The modal reference
    impedances ``z_modal`` are real constants computed from the transform, or
    given with `modal_z0`. As for the choice of `z0` in the mixed-mode class,
    the split depends weakly on them: on synthetic lines, a reference 10 %
    away from the modal impedance of the lines changed the de-embedded DUT
    by about 1e-3 in S-parameter units.

    ``modal_transform`` can be:

    ``'eig_const'`` (default)
        One constant, real transform estimated from the eigenvectors of the
        2x-thru, which are tracked over frequency and averaged with the
        relative eigenvalue spacing as weight. This is much less sensitive to
        noise than ``'eig'``. It needs modes with different propagation delays or
        losses; it cannot find the transform of lines in a homogeneous medium
        with equal losses, in which case the transform should be supplied.
    ``'eig'``
        One transform for each frequency from the eigenvectors of the 2x-thru
        (tracked over frequency, unit-norm columns, or the image impedance
        scaling with ``eig_scaling='image'``). It is fragile with measured
        data: near-equal eigenvalues (at low frequency, and wherever the modes
        have the same electrical length modulo a half-wave) make the vectors
        noisy. A :class:`RuntimeWarning` is issued if they are not smooth.
    ``'mixed'``
        The differential and common mode transform, N = 2 only. Exactly
        :class:`IEEEP370_MM_NZC_2xThru`.
    array
        A transform of shape (N, N), or (F, N, N) for one transform per
        frequency, columns being the modes. Use it when the modes are known, for
        example from a field solver or from the line geometry.

    Port numbering follows the `port_order` of
    :class:`IEEEP370_MM_NZC_2xThru`, extended to N lines.

    numbering diagram, N = 3 lines::

      port_order = 'first'          port_order = 'second'
           +---------+                   +---------+
          -|0       1|-                 -|0       3|-
          -|2       3|-                 -|1       4|-
          -|4       5|-                 -|2       5|-
           +---------+                   +---------+

      port_order = 'third'
           +---------+
          -|0       5|-
          -|1       4|-
          -|2       3|-
           +---------+

    The `scikit-rf` cascade ** 2N-port operator uses the second scheme, which
    is used internally. The error boxes ``se_side1`` and ``se_side2`` are
    always in this scheme, whatever `port_order` is.

    Example
    -------
    >>> import skrf as rf
    >>> from skrf.calibration.deembedding import Modal_NZC_2xThru

    Create network objects for 2x-thru and FIX-DUT-FIX, here with 6 ports (3 lines)

    >>> s2xthru = rf.Network('2xthru.s6p')
    >>> fdf = rf.Network('f-dut-f.s6p')

    Create de-embedding object

    >>> dm = Modal_NZC_2xThru(dummy_2xthru = s2xthru, name = '2xthru')

    Apply deembedding to get the actual DUT network

    >>> dut = dm.deembed(fdf)

    Note
    ----
    numbering diagram, port_order = 'second', N = 2 lines::

         FIX-1    DUT      FIX-2
         +----+   +----+   +----+
        -|0  2|---|0  2|---|2  0|-
        -|1  3|---|1  3|---|3  1|-
         +----+   +----+   +----+

    Attributes
    ----------
    se_side1, se_side2 : :class:`~skrf.network.Network`
        Left and right error boxes, 'second' port order.
    tv : :class:`numpy.ndarray`
        Modal transform used, shape (F, N, N).
    z_modal : :class:`numpy.ndarray`
        Modal reference impedance of every mode, shape (N,), real.
    modal_2xthru : :class:`~skrf.network.Network`
        The 2x-thru in the modal basis.
    mode_order : :class:`numpy.ndarray`
        Order of the modes of the eigenvector options: slowest first,
        by the real part of the eigenvalue at the lowest frequency.
    x_end, z_x : list
        Debug outputs of :class:`IEEEP370_SE_NZC_2xThru`, one entry for each mode.
    mode_conversion_db, transform_ripple, mirror_asymmetry, eig_gap :
        Diagnostics, see above. ``None`` when they do not apply.

    References
    ----------
    .. [ElSA20] Ellison J, Smith SB, Agili S., "Using a 2x-thru standard to achieve
        accurate de-embedding of measurements", Microwave Optical Technology
        Letter, 2020, https://doi.org/10.1002/mop.32098
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370mmZc2xthru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    """

    #: warn if the modal vectors deviate more than this from a cubic fit in frequency (``'eig'``)
    transform_ripple_limit = 1e-2
    #: warn if the 2x-thru is farther than this from the mirror image of itself, relative. Noise of
    #: 2e-4 in magnitude and 0.2 degree in phase gives about 1e-2 on synthetic data, a fixture that
    #: is not its own mirror image gives 0.5 or more.
    mirror_asymmetry_limit = 5e-2
    #: warn if the relative eigenvalue spacing is below this over most of the band
    eig_gap_limit = 1e-4

    _TRANSFORMS = ('mixed', 'eig_const', 'eig')

    def __init__(self, dummy_2xthru: Network, name: str = None,
                 z0: float = 50, port_order: PortOrderT = 'second',
                 modal_transform: str | ndarray = 'eig_const', modal_z0: Sequence[float] = None,
                 eig_scaling: Literal['unit', 'image'] = 'unit',
                 use_z_instead_ifft: bool = False, forced_z0_line: Sequence[float] = None,
                 conversion_limit_db: float = -30.0, verbose: bool = False,
                 *args, **kwargs) -> None:
        """
        Modal_NZC_2xThru De-embedding Initializer

        Parameters
        -----------

        dummy_2xthru : :class:`~skrf.network.Network` object
            2x-thru (FIX-FIX) network with 2N ports. It is expected to be
            reciprocal and the mirror image of itself, and to have a harmonic
            frequency axis, like for the single-ended method.

        name : string
            optional name of de-embedding object

        z0 : float
            real reference impedance of the S-parameters, the same for all
            the ports (default: 50). The 2x-thru is renormalized (a copy) if
            it has another reference impedance.

        port_order : ['first', 'second', 'third']
            specify what numbering scheme to use. See above. (default: second)

        modal_transform : ['eig_const', 'eig', 'mixed'] or array
            how the modal transform is obtained, see above. (default: 'eig_const')

        modal_z0 : array-like of N reals, optional
            modal reference impedances, overriding the ones computed from the
            modal transform. They have the same role as the factors 2 * z0
            (differential) and z0 / 2 (common) of the mixed-mode class.

        eig_scaling : ['unit', 'image']
            column scaling of the per-frequency eigenvectors, only for
            ``modal_transform='eig'``: unit norm, or constant modal image
            impedance. Neither is universally better: unit-norm columns
            are wrong if the true transform depends on frequency, and the image
            impedance scaling is biased by connectors. (default: 'unit')

        use_z_instead_ifft :
            use z-transform instead ifft, for each mode. See
            :class:`IEEEP370_SE_NZC_2xThru`. (default: False)

        forced_z0_line : array-like of N reals, optional
            If specified, the split plane impedance of each mode is forced to
            these values, in modal ohms, in the order of the columns of the
            transform (``mode_order`` for the eigenvector options). An entry
            None leaves the mode free. It has the same meaning as
            `forced_z0_line_dd` and `forced_z0_line_cc` of
            :class:`IEEEP370_MM_NZC_2xThru`, which are the entries for the
            differential and the common modes. (Default: None)

        conversion_limit_db : float
            limit of ``mode_conversion_db`` in dB above which a warning is
            issued. FER6 of IEEE 370 uses -15 dB for the differential to common
            conversion of a 2x-thru, but the conversion left in the estimated
            modal basis reached only -24 dB on synthetic fixtures with a
            mode-converting launch, where the de-embedding error was already
            0.14 (S-parameter units), so the default is stricter. (default: -30)

        verbose :
            view the process (default: False)

        args, kwargs:
            passed to :func:`Deembedding.__init__`

        Raises
        ------
        ValueError
            for an odd number of ports, an invalid option, a transform that is
            not invertible or does not fit the 2x-thru.

        See Also
        ---------
        :func:`Deembedding.__init__`
        :class:`IEEEP370_MM_NZC_2xThru`

        """
        self.s2xthru = dummy_2xthru.copy()
        nports = self.s2xthru.nports
        if nports % 2 != 0 or nports < 2:
            raise ValueError('2xthru must have an even number of ports (2N ports for N lines), '
                             f'it has {nports}.')
        self.n_lines = n = nports // 2

        z0_arr = np.asarray(z0)
        if z0_arr.ndim != 0 or np.imag(z0_arr) != 0 or not np.real(z0_arr) > 0:
            raise ValueError('z0 must be a positive real scalar.')
        self.z0 = float(np.real(z0_arr))
        if np.any(np.abs(self.s2xthru.z0 - self.z0) > 1e-9 * self.z0):
            self.s2xthru.renormalize(self.z0)

        if port_order not in ('first', 'second', 'third'):
            raise ValueError(f"port_order must be 'first', 'second' or 'third', not {port_order!r}.")
        self.port_order = port_order

        self._tv_user = None
        if isinstance(modal_transform, str):
            if modal_transform not in self._TRANSFORMS:
                raise ValueError(f'modal_transform must be one of {self._TRANSFORMS} or an array, '
                                 f'not {modal_transform!r}.')
            if modal_transform == 'mixed' and n != 2:
                raise ValueError("modal_transform='mixed' is for differential pairs only (4 ports).")
        else:
            tv = np.asarray(modal_transform)
            if tv.ndim == 2:
                tv = tv[None]
            if tv.ndim != 3 or tv.shape[1:] != (n, n):
                raise ValueError(f'modal_transform array must have shape ({n}, {n}) or (F, {n}, {n}), '
                                 f'not {np.shape(modal_transform)}.')
            if tv.shape[0] not in (1, self.s2xthru.frequency.npoints):
                raise ValueError(f'modal_transform array has {tv.shape[0]} frequency points, '
                                 f'the 2x-thru has {self.s2xthru.frequency.npoints}.')
            if not np.all(np.isfinite(tv)) or np.any(np.linalg.cond(tv) > 1.0 / np.finfo(float).eps):
                raise ValueError('modal_transform is not invertible.')
            self._tv_user = tv.astype(complex)
            modal_transform = 'user'
        self.modal_transform = modal_transform

        if eig_scaling not in ('unit', 'image'):
            raise ValueError(f"eig_scaling must be 'unit' or 'image', not {eig_scaling!r}.")
        if eig_scaling == 'image' and modal_transform != 'eig':
            raise ValueError("eig_scaling='image' applies to modal_transform='eig' only.")
        self.eig_scaling = eig_scaling

        if modal_z0 is not None:
            modal_z0 = np.asarray(modal_z0, dtype=float)
            if modal_z0.shape != (n,) or not np.all(modal_z0 > 0):
                raise ValueError(f'modal_z0 must be {n} positive real impedances.')
        self.modal_z0 = modal_z0

        if forced_z0_line is not None:
            forced_z0_line = list(np.atleast_1d(forced_z0_line))
            if len(forced_z0_line) != n:
                raise ValueError(f'forced_z0_line must have one entry for each of the {n} modes, '
                                 f'it has {len(forced_z0_line)}.')
        self.forced_z0_line = forced_z0_line

        self.conversion_limit_db = conversion_limit_db
        self.use_z_instead_ifft = use_z_instead_ifft
        self.verbose = verbose
        dummies = [self.s2xthru]

        # debug outputs and diagnostics
        self.x_end = None
        self.z_x = None
        self.tv = None
        self.z_modal = None
        self.modal_2xthru = None
        self.mode_order = None
        self.eigenvalues = None
        self.mode_conversion_db = None
        self.transform_ripple = None
        self.mirror_asymmetry = None
        self.eig_gap = None

        IEEEP370.__init__(self, dummies, name, *args, **kwargs)
        self.se_side1, self.se_side2 = self.split2xthru(self.s2xthru)

    def _port_permutation(self, nports: int) -> list[int] | None:
        """
        Position of the 'second' ports in the `port_order` numbering, None for 'second'.
        """
        if self.port_order == 'first':
            return list(range(0, nports, 2)) + list(range(1, nports, 2))
        if self.port_order == 'third':
            return list(range(0, nports // 2)) + list(range(nports - 1, nports // 2 - 1, -1))
        return None

    def _to_second(self, ntwk: Network) -> Network:
        """
        Copy of `ntwk` renumbered from `port_order` to the 'second' order.
        """
        perm = self._port_permutation(ntwk.nports)
        if perm is None:
            return ntwk.copy()
        return ntwk.renumbered(perm, list(range(ntwk.nports)))

    def _from_second(self, ntwk: Network) -> Network:
        """
        Copy of `ntwk` renumbered from the 'second' order to `port_order`.
        """
        perm = self._port_permutation(ntwk.nports)
        if perm is None:
            return ntwk.copy()
        return ntwk.renumbered(list(range(ntwk.nports)), perm)

    def deembed(self, ntwk: Network) -> Network:
        """
        Perform the de-embedding calculation

        Parameters
        ----------
        ntwk : :class:`~skrf.network.Network` object
            FIX-DUT-FIX network from which FIX-1 and FIX-2 fixtures needs
            to be removed via de-embedding. It is not modified.

        Returns
        -------
        caled : :class:`~skrf.network.Network` object
            Network data of the device after de-embedding, with the port
            numbering of `port_order`.

        """
        if ntwk.nports != self.s2xthru.nports:
            raise ValueError(f'The network has {ntwk.nports} ports, the 2xthru has {self.s2xthru.nports}.')

        # check if the frequencies match with dummy frequencies
        if ntwk.frequency != self.s2xthru.frequency:
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.',
                          RuntimeWarning, stacklevel=2)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            se_side1, se_side2 = self.split2xthru(s2xthru)
        else:
            se_side1 = self.se_side1
            se_side2 = self.se_side2

        ntwk = ntwk.copy()
        if np.any(np.abs(ntwk.z0 - self.z0) > 1e-9 * self.z0):
            ntwk.renormalize(self.z0)
        ntwk = self._to_second(ntwk)
        deembedded = se_side1.inv ** ntwk ** se_side2.flipped().inv
        return self._from_second(deembedded)

    def _get_transform(self, s2x: Network) -> tuple[ndarray, dict]:
        """
        Modal transform, shape (F, N, N), and information of the eigenvector options.
        """
        n_f = s2x.frequency.npoints
        if self.modal_transform == 'mixed':
            return np.broadcast_to(_MIXED_MODE_TV, (n_f, 2, 2)).astype(complex), {}
        if self.modal_transform == 'user':
            tv = self._tv_user
            if tv.shape[0] not in (1, n_f):
                raise ValueError(f'modal_transform has {tv.shape[0]} frequency points, the 2x-thru has {n_f}.')
            return np.broadcast_to(tv, (n_f, self.n_lines, self.n_lines)).astype(complex), {}
        mode = 'freq' if self.modal_transform == 'eig' else 'const'
        try:
            return _eig_modal_transform(s2x, self.z0, mode, self.eig_scaling)
        except np.linalg.LinAlgError as err:
            raise ValueError('Modal 2x-thru: the S21 block of the 2x-thru is singular at some frequency, '
                             'its ABCD matrix does not exist.') from err

    def split2xthru(self, se_2xthru: Network) -> (Network, Network):
        """
        Perform the fixtures extraction.

        Parameters
        ----------
        se_2xthru : :class:`~skrf.network.Network` object
            2x-thru with the numbering of `port_order`. It is not modified.

        Returns
        -------
        se_side1, se_side2 : :class:`~skrf.network.Network`
            FIX-1 and FIX-2 error boxes in 'second' order, FIX-2 being
            the mirror image, like for :class:`IEEEP370_MM_NZC_2xThru`.
        """
        n = self.n_lines
        if se_2xthru.nports != 2 * n:
            raise ValueError(f'2xthru has to be a {2 * n}-port network.')
        s2x = self._to_second(se_2xthru)
        frequency = s2x.frequency
        n_f = frequency.npoints

        tv, info = self._get_transform(s2x)
        z_modal = self.modal_z0 if self.modal_z0 is not None else _modal_references(tv, self.z0)
        z_modal = np.asarray(z_modal, dtype=float)
        xi = _modal_xi(tv, self.z0, z_modal)
        try:
            s_m = _se2modal(s2x.s, xi)
        except np.linalg.LinAlgError as err:
            raise ValueError('Modal 2x-thru: the modal transformation of the 2x-thru is singular.') from err
        self.tv = tv
        self.z_modal = z_modal
        self.modal_2xthru = Network(frequency=frequency, s=s_m,
                                    z0=np.broadcast_to(np.concatenate([z_modal, z_modal]), (n_f, 2 * n)).copy())
        self.eigenvalues = info.get('eigenvalues')
        self.mode_order = info.get('mode_order')
        self._check_line_likeness(s2x, s_m, info)

        # split every mode, a single-ended 2-port between ports k and N + k
        s_1 = np.zeros_like(s_m)
        s_2 = np.zeros_like(s_m)
        x_end = []
        z_x = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for k in range(n):
                idx = [k, n + k]
                two = Network(frequency=frequency, s=s_m[:, idx][:, :, idx], z0=z_modal[k])
                forced = None if self.forced_z0_line is None else self.forced_z0_line[k]
                dm = IEEEP370_SE_NZC_2xThru(dummy_2xthru=two, z0=float(z_modal[k]),
                                            use_z_instead_ifft=self.use_z_instead_ifft,
                                            verbose=self.verbose, forced_z0_line=forced)
                x_end.append(dm.x_end)
                z_x.append(dm.z_x)
                for i in range(2):
                    for j in range(2):
                        s_1[:, idx[i], idx[j]] = dm.s_side1.s[:, i, j]
                        s_2[:, idx[i], idx[j]] = dm.s_side2.s[:, i, j]
        # the single-ended method warns for every mode, say it once
        seen = set()
        for w in caught:
            key = (w.category, str(w.message))
            if key not in seen:
                seen.add(key)
                warnings.warn(w.message, w.category, stacklevel=3)
        self.x_end = x_end
        self.z_x = z_x

        # back to terminal S-parameters
        se_side1 = Network(frequency=frequency, s=_modal2se(s_1, xi), z0=self.z0)
        se_side2 = Network(frequency=frequency, s=_modal2se(s_2, xi), z0=self.z0)
        return (se_side1, se_side2)

    def _check_line_likeness(self, s2x: Network, s_m: ndarray, info: dict) -> None:
        """
        Compute the diagnostics and warn if the 2x-thru does not look like N uniform coupled lines.
        """
        n = self.n_lines
        f = s2x.f
        prefix = 'Modal 2x-thru: '

        # mode conversion in the modal basis, like FER6
        mag_db = 20.0 * np.log10(np.abs(s_m) + 1e-300)
        conversion = np.full(len(f), -np.inf)
        for k in range(n):
            for j in range(n):
                if j != k:
                    conversion = np.maximum(conversion, mag_db[:, j, n + k] - mag_db[:, k, n + k])
        self.mode_conversion_db = conversion
        worst = int(np.argmax(conversion))
        if conversion[worst] > self.conversion_limit_db:
            warnings.warn(f'{prefix}the mode conversion of the 2x-thru reaches {conversion[worst]:.1f} dB '
                          f'relative to the insertion loss of the mode at {f[worst] / 1e9:.4g} GHz, above the '
                          f'limit of {self.conversion_limit_db:g} dB. The fixture does not look like uniform '
                          'coupled lines in this modal basis, the splitting of each mode may be inaccurate.',
                          RuntimeWarning, stacklevel=3)

        # mirror symmetry, A = D^T for a reciprocal 2x-thru that is its own mirror image
        valid = f > 0
        try:
            abcd = _abcd_2nport(s2x.s[valid], self.z0)
        except np.linalg.LinAlgError as err:
            raise ValueError('Modal 2x-thru: the S21 block of the 2x-thru is singular at some frequency, '
                             'its ABCD matrix does not exist.') from err
        block_a = abcd[:, :n, :n]
        block_d = abcd[:, n:, n:]
        asym = np.linalg.norm(block_a - np.swapaxes(block_d, 1, 2), axis=(1, 2)) / np.linalg.norm(block_a, axis=(1, 2))
        self.mirror_asymmetry = float(np.max(asym))
        if self.mirror_asymmetry > self.mirror_asymmetry_limit:
            warnings.warn(f'{prefix}the 2x-thru is not the mirror image of itself, the relative difference '
                          f'between the blocks A and D^T of its ABCD matrix is {self.mirror_asymmetry:.2g} '
                          f'(limit {self.mirror_asymmetry_limit:g}). The splitting assumes FIX-2 is the mirror '
                          'of FIX-1.', RuntimeWarning, stacklevel=3)

        if self.modal_transform not in ('eig', 'eig_const'):
            self.transform_ripple = None
            self.eig_gap = None
            return
        self.transform_ripple = info['ripple']
        self.eig_gap = info['eig_gap']
        if self.modal_transform == 'eig' and info['ripple'] > self.transform_ripple_limit:
            warnings.warn(f'{prefix}the eigenvectors of the 2x-thru deviate by {info["ripple"]:.2g} from a smooth '
                          f'function of frequency (limit {self.transform_ripple_limit:g}). Either the data are '
                          'noisy, or the fixture does not look like uniform coupled lines. Consider '
                          "modal_transform='eig_const' or a transform given as an array.",
                          RuntimeWarning, stacklevel=3)
        if np.mean(info['eig_gap'][valid] < self.eig_gap_limit) > 0.5:
            warnings.warn(f'{prefix}the modes of the 2x-thru have nearly equal eigenvalues over most of the band '
                          f'(relative spacing below {self.eig_gap_limit:g}), as in a homogeneous medium. The '
                          'modal transform cannot be found from the 2x-thru, supply it as an array or use '
                          "modal_transform='mixed' for a differential pair.",
                          RuntimeWarning, stacklevel=3)
