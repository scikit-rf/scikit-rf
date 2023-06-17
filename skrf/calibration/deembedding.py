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
   IEEEP370_SE_NZC_2xThru
   IEEEP370_MM_NZC_2xThru
   IEEEP370_SE_ZC_2xThru
   IEEEP370_MM_ZC_2xThru

"""

from abc import ABC, abstractmethod
from ..frequency import *
from ..network import *
import warnings
import numpy as np
from numpy import concatenate, conj, flip, real, angle, exp, zeros
from numpy.fft import fft, fftshift, irfft, ifftshift
from scipy.interpolate import interp1d

from ..util import subplots

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
                warnings.warn('Dummy Networks dont have matching frequencies, attempting overlap.', RuntimeWarning)
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

        output = '%s Deembedding: %s, %s, %s dummy structures'\
                %(self.__class__.__name__, name, str(self.frequency),\
                    len(self.dummies))

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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
            ntwk, thru = overlap_multi([ntwk, self.thru])
        else:
            thru = self.thru

        caled = ntwk.copy()
        h = ntwk ** thru.inv
        h_ = h.flipped()
        caled.z = (h.z + h_.z) / 2

        return caled


class IEEEP370_SE_NZC_2xThru(Deembedding):
    """
    Creates error boxes from a test fixture 2xThru network.

    Based on [ElSA20]_ and [I3E370]_.

    A deembedding object is created with a single 2xThru (FIX-FIX) network,
    which is split into left (FIX-1) and right (FIX-2) fixtures with IEEEP370
    2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1 and
    FIX-2 are deembedded from the FIX_DUT_FIX network.

    This method is applicable only when there is a 2x-Thru network.

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
    def __init__(self, dummy_2xthru, name=None,
                 z0 = 50, use_z_instead_ifft = False, verbose = False,
                 forced_z0_line = None, *args, **kwargs):
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

        Deembedding.__init__(self, dummies, name, *args, **kwargs)
        self.s_side1, self.s_side2 = self.split2xthru(self.s2xthru)

    def deembed(self, ntwk):
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            s_side1, s_side2 = self.split2xthru(s2xthru)
        else:
            s_side1 = self.s_side1
            s_side2 = self.s_side2

        return s_side1.inv ** ntwk ** s_side2.flipped().inv

    def dc_interp(self, s, f):
        """
        enforces symmetric upon the first 10 points and interpolates the DC
        point.
        """
        sp = s[0:9]
        fp = f[0:9]

        snp = concatenate((conj(flip(sp)), sp))
        fnp = concatenate((-1*flip(fp), fp))
        # mhuser : used cubic instead spline (not implemented)
        snew = interp1d(fnp, snp, axis=0, kind = 'cubic')
        return real(snew(0))

    def COM_receiver_noise_filter(self, f,fr):
        """
        receiver filter in COM defined by eq 93A-20
        """
        fdfr = f / fr
        # eq 93A-20
        return 1 / (1 - 3.414214 * fdfr**2 + fdfr**4 + 1j*2.613126*(fdfr - fdfr**3))


    def makeStep(self, impulse):
        #mhuser : no need to call step function here, cumsum will be enough and efficient
        #step = np.convolve(np.ones((len(impulse))), impulse)
        #return step[0:len(impulse)]
        return np.cumsum(impulse, axis=0)


    def makeSymmetric(self, nonsymmetric):
        """
        this takes the nonsymmetric frequency domain input and makes it
        symmetric.
        The function assumes the DC point is in the nonsymmetric data
        """
        symmetric_abs = concatenate((np.abs(nonsymmetric), flip(np.abs(nonsymmetric[1:]))))
        symmetric_ang = concatenate((angle(nonsymmetric), -flip(angle(nonsymmetric[1:]))))
        return symmetric_abs * exp(1j * symmetric_ang)

    def DC(self, s, f):
        DCpoint = 0.002 # seed for the algorithm
        err = 1 # error seed
        allowedError = 1e-12 # allowable error
        cnt = 0
        df = f[1] - f[0]
        n = len(f)
        t = np.linspace(-1/df,1/df,n*2+1)
        ts = np.argmin(np.abs(t - (-3e-9)))
        Hr = self.COM_receiver_noise_filter(f, f[-1]/2)
        while(err > allowedError):
            h1 = self.makeStep(fftshift(irfft(concatenate(([DCpoint], Hr * s)), axis=0), axes=0))
            h2 = self.makeStep(fftshift(irfft(concatenate(([DCpoint + 0.001], Hr * s)), axis=0), axes=0))
            m = (h2[ts] - h1[ts]) / 0.001
            b = h1[ts] - m * DCpoint
            DCpoint = (0 - b) / m
            err = np.abs(h1[ts] - 0)
            cnt += 1
        return DCpoint


    def split2xthru(self, s2xthru):
        f = s2xthru.frequency.f
        s = s2xthru.s

        if not self.use_z_instead_ifft:
            # strip DC point if one exists
            if(f[0] == 0):
                warnings.warn(
                    "DC point detected. An interpolated DC point will be included in the errorboxes.",
                    RuntimeWarning
                    )
                flag_DC = True
                f = f[1:]
                s = s[1:]
            else:
                flag_DC = False

            # interpolate S-parameters if the frequency vector is not acceptable
            if(f[1] - f[0] != f[0]):
                warnings.warn(
                   """Non-uniform frequency vector detected. An interpolated S-parameter matrix will be created for this calculation.
                   The output results will be re-interpolated to the original vector.""",
                   RuntimeWarning
                   )
                flag_df = True
                f_original = f
                df = f[1] - f[0]
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
            dcs21 = self.dc_interp(s21, f)
            t21 = fftshift(irfft(concatenate(([dcs21], s21)), axis=0), axes=0)
            x = np.argmax(t21)

            dcs11 = self.DC(s11,f)
            t11 = fftshift(irfft(concatenate(([dcs11], s11)), axis=0), axes=0)
            step11 = self.makeStep(t11)
            z11 = -self.z0 * (step11 + 1) / (step11 - 1)

            if self.forced_z0_line:
                z11x = self.forced_z0_line
            else:
                z11x = z11[x]

            if self.verbose:
                fig, (ax1, ax2) = subplots(2,1)
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

            dcs11r = self.DC(s11r, f)
            # irfft is equivalent to ifft(makeSymmetric(x))
            t11r = fftshift(irfft(concatenate(([dcs11r], s11r)), axis=0), axes=0)
            t11r[x:] = 0
            e001 = fft(ifftshift(t11r))
            e001 = e001[1:n+1]

            dcs22r = self.DC(s22r, f)
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
                    if npy.abs(-e01[i] - e01[i-1]) < npy.abs(e01[i] - e01[i-1]):
                        e01[i] = - e01[i]
            e10 = np.sqrt(s12r * (1 - e111 * e112))
            for i in range(n):
                if i > 0:
                    if npy.abs(-e10[i] - e10[i-1]) < npy.abs(e10[i] - e10[i-1]):
                        e10[i] = - e10[i]


            # revert to initial freq axis
            if flag_df:
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
                e001 = concatenate(([self.dc_interp(e001, f)], e001))
                e01  = concatenate(([self.dc_interp(e01, f)], e01))
                e111 = concatenate(([self.dc_interp(e111, f)], e111))
                e002 = concatenate(([self.dc_interp(e002, f)], e002))
                e10 = concatenate(([self.dc_interp(e10, f)], e10))
                e112 = concatenate(([self.dc_interp(e112, f)], e112))
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

class IEEEP370_MM_NZC_2xThru(Deembedding):
    """
    Creates error boxes from a 4-port test fixture 2xThru.

    Based on [ElSA20]_ and [I3E370]_.

    A deembedding object is created with a single 2xThru (FIX-FIX) network,
    which is split into left (FIX-1_2) and right (FIX-3_4) fixtures with
    IEEEP370 2xThru method.

    When :func:`Deembedding.deembed` is applied, the s-parameters of FIX-1 and
    FIX-2 are deembedded from the FIX_DUT_FIX network.

    This method is applicable only when there is a 2x-Thru measurement.

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
    [I3E370]:
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
    def __init__(self, dummy_2xthru, name=None,
                 z0 = 50, port_order: str = 'second',
                 use_z_instead_ifft = False, verbose = False,
                 forced_z0_line_dd = None, forced_z0_line_cc = None, *args, **kwargs):
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
        self.forced_z0_line_dd = forced_z0_line_dd
        self.forced_z0_line_cc = forced_z0_line_cc
        self.verbose = verbose

        Deembedding.__init__(self, dummies, name, *args, **kwargs)
        self.se_side1, self.se_side2 = self.split2xthru(self.s2xthru)

    def deembed(self, ntwk):
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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

    def split2xthru(self, se_2xthru):
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
        dm_cc  = IEEEP370_SE_NZC_2xThru(dummy_2xthru = scc, z0 = self.z0 / 2,
                                use_z_instead_ifft = self.use_z_instead_ifft,
                                verbose = self.verbose,
                                forced_z0_line = self.forced_z0_line_cc)

        #convert back to single-ended
        mm_side1 = concat_ports([dm_dd.s_side1, dm_cc.s_side1], port_order = 'first')
        se_side1 = mm_side1.copy()
        se_side1.gmm2se(p = 2)
        mm_side2 = concat_ports([dm_dd.s_side2, dm_cc.s_side2], port_order = 'first')
        se_side2 = mm_side2.copy()
        se_side2.gmm2se(p = 2)

        return (se_side1, se_side2)


class IEEEP370_SE_ZC_2xThru(Deembedding):
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
    [I3E370]:
        - FIX-2 is flipped (see diagram above)

    References
    ----------
    .. [I3E370] https://opensource.ieee.org/elec-char/ieee-370/-/blob/master/TG1/IEEEP370Zc2xThru.m
       commit 49ddd78cf68ad5a7c0aaa57a73415075b5178aa6
    """
    def __init__(self, dummy_2xthru, dummy_fix_dut_fix, name=None,
                 z0 = 50, bandwidth_limit = 0,
                 pullback1 = 0, pullback2 = 0,
                 side1 = True, side2 = True,
                 NRP_enable = True, leadin = 1,
                 verbose = False,
                 *args, **kwargs):
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

        Deembedding.__init__(self, dummies, name, *args, **kwargs)
        self.s_side1, self.s_side2 = self.split2xthru(self.s2xthru.copy(),
                                                      self.sfix_dut_fix)

    def deembed(self, ntwk):
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
            ntwk, s2xthru = overlap_multi([ntwk, self.s2xthru])
            s_side1, s_side2 = self.split2xthru(s2xthru,
                                                      self.sfix_dut_fix)
        else:
            s_side1 = self.s_side1
            s_side2 = self.s_side2

        return s_side1.inv ** ntwk ** s_side2.flipped().inv

    def thru(self, n):
        out = n.copy()
        out.s[:, 0, 0] = 0
        out.s[:, 1, 0] = 1
        out.s[:, 0, 1] = 1
        out.s[:, 1, 1] = 0
        return out

    def add_dc(self, sin):
        s = sin.s
        f = sin.frequency.f

        n = len(f)
        snew = zeros((n + 1, 2,2), dtype = complex)
        snew[1:,:,:] = s
        snew[0, 0, 0] = self.dc_interp(s[:, 0, 0], f)
        snew[0, 0, 1] = self.dc_interp(s[:, 0, 1], f)
        snew[0, 1, 0] = self.dc_interp(s[:, 1, 0], f)
        snew[0, 1, 1] = self.dc_interp(s[:, 1, 1], f)

        f = concatenate(([0], f))
        return Network(frequency = Frequency.from_f(f, 'Hz'), s = snew)

    def dc_interp(self, s, f):
        """
        enforces symmetric upon the first 10 points and interpolates the DC
        point.
        """
        sp = s[0:9]
        fp = f[0:9]

        snp = concatenate((conj(flip(sp)), sp))
        fnp = concatenate((-1*flip(fp), fp))
        # mhuser : used cubic instead spline (not implemented)
        snew = interp1d(fnp, snp, axis=0, kind = 'cubic')
        return real(snew(0))

    def COM_receiver_noise_filter(self, f,fr):
        """
        receiver filter in COM defined by eq 93A-20
        """
        f = f / 1e9
        fdfr = f / fr
        # eq 93A-20
        return 1 / (1 - 3.414214 * fdfr**2 + fdfr**4 + 1j*2.613126*(fdfr - fdfr**3))


    def makeStep(self, impulse):
        #mhuser : no need to call step function here, cumsum will be enough and efficient
        #step = np.convolve(np.ones((len(impulse))), impulse)
        #return step[0:len(impulse)]
        return np.cumsum(impulse, axis=0)

    def DC2(self, s, f):
        DCpoint = 0.002 # seed for the algorithm
        err = 1 # error seed
        allowedError = 1e-10 # allowable error
        cnt = 0
        df = f[1] - f[0]
        n = len(f)
        t = np.linspace(-1/df,1/df,n*2+1)
        ts = np.argmin(np.abs(t - (-3e-9)))
        Hr = self.COM_receiver_noise_filter(f, f[-1]/2)
        while(err > allowedError):
            h1 = self.makeStep(fftshift(irfft(concatenate(([DCpoint], Hr * s)), axis=0), axes=0))
            h2 = self.makeStep(fftshift(irfft(concatenate(([DCpoint + 0.001], Hr * s)), axis=0), axes=0))
            m = (h2[ts] - h1[ts]) / 0.001
            b = h1[ts] - m * DCpoint
            DCpoint = (0 - b) / m
            err = np.abs(h1[ts] - 0)
            cnt += 1
        return DCpoint

    def getz(self, s, f, z0):
        DC11 = self.DC2(s, f)
        t112x = irfft(concatenate(([DC11], s)))
        #get the step response of t112x. Shift is needed for makeStep to
        #work properly.
        t112xStep = self.makeStep(fftshift(t112x))
        #construct the transmission line
        z = -z0 * (t112xStep + 1) / (t112xStep - 1)
        z = ifftshift(z) #impedance. Shift again to get the first point
        return z[0], z

    def makeTL(self, zline, z0, gamma, l):
        # todo: use DefinedGammaZ0 media instead
        n = len(gamma)
        TL = np.zeros((n, 2, 2), dtype = complex)
        TL[:, 0, 0] = ((zline**2 - z0**2) * np.sinh(gamma * l)) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        TL[:, 1, 0] = (2 * z0 * zline) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        TL[:, 0, 1] = (2 * z0 * zline) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        TL[:, 1, 1] = ((zline**2 - z0**2) * np.sinh(gamma * l)) / ((zline**2 + z0**2) * np.sinh(gamma * l) + 2 * z0 * zline * np.cosh(gamma * l))
        return TL

    def NRP(self, nin, TD = None, port = None):
        p = nin.s
        f = nin.frequency.f
        n = len(f)
        X = nin.nports
        fend = f[-1]
        if TD is None:
            TD = np.zeros(X)
            for i in range(X):
                theta0 = np.angle(p[-1, i, i])
                if theta0 < -np.pi/2:
                    theta = -np.pi - theta0
                elif theta0 > np.pi/2:
                    theta = np.pi - theta0
                else:
                    theta = -theta0
                TD[i] = -theta / (2 * np.pi * fend)
                pd = np.zeros((n, X, X), dtype = complex)
                delay = np.exp(-1j * 2. * np.pi * f * TD[i] / 2.)
                if i == 0:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = nin.copy()
                    spd.s = pd
                    out = spd ** nin
                elif i < X//2:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = nin.copy()
                    spd.s = pd
                    out = spd ** out
                else:
                    pd[:, i - X//2, i] = delay
                    pd[:, i, i - X//2] = delay
                    spd = nin.copy()
                    spd.s = pd
                    out = out ** spd
        else:
            pd = np.zeros((n, X, X), dtype = complex)
            if port is not None:
                i = port
                delay = np.exp(1j * 2. * np.pi * f * TD[i] / 2.)
                if i < X//2:
                    pd[:, i + X//2, i] = delay
                    pd[:, i, i + X//2] = delay
                    spd = nin.copy()
                    spd.s = pd
                    out = spd ** nin
                else:
                    pd[:, i - X//2, i] = delay
                    pd[:, i, i - X//2] = delay
                    spd = nin.copy()
                    spd.s = pd
                    out = nin ** spd
            else:
                for i in range(X):
                    delay = np.exp(1j * 2. * np.pi * f * TD[i] / 2)
                    if i == 0:
                        pd[:, i + X//2, i] = delay
                        pd[:, i, i + X//2] = delay
                        spd = nin.copy()
                        spd.s = pd
                        out = spd ** nin
                    elif i < X//2:
                        pd[:, i + X//2, i] = delay
                        pd[:, i, i + X//2] = delay
                        spd = nin.copy()
                        spd.s = pd
                        out = spd ** out
                    else:
                        pd[:, i - X//2, i] = delay
                        pd[:, i, i - X//2] = delay
                        spd = nin.copy()
                        spd.s = pd
                        out = out ** spd
        return out, TD

    def shiftOnePort(self, nin, N, port):
        f = nin.frequency.f
        n = len(f)
        X = nin.nports
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        delay = np.exp(-N * 1j * Omega/2)
        pd = np.zeros((n, 2, 2), dtype = complex)
        if port < X//2:
            pd[:, port, port + X//2] = delay
            pd[:, port + X//2, port] = delay
            spd = nin.copy()
            spd.s = pd
            out = spd ** nin
        else:
            pd[:, port, port - X//2] = delay
            pd[:, port - X//2, port] = delay
            spd = nin.copy()
            spd.s = pd
            out = nin ** spd
        return out

    def shiftNPoints(self, nin, N):
        f = nin.frequency.f
        n = len(f)
        X = nin.nports
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        delay = np.exp(-N * 1j * Omega/2)
        pd = np.zeros((n, 2, 2), dtype = complex)
        for port in range(X):
            if port < X//2:
                pd[:, port, port + X//2] = delay
                pd[:, port + X//2, port] = delay
            else:
                pd[:, port, port - X//2] = delay
                pd[:, port - X//2, port] = delay
                spd = nin.copy()
                spd.s = pd
                out = nin ** spd
        return out

    def peelNPointsLossless(self, nin, N):
        f = nin.frequency.f
        n = len(f)
        out = nin.copy()
        z0 = 50
        Omega0 = np.pi/n
        Omega = np.arange(Omega0, np.pi + Omega0, Omega0)
        betal = 1j * Omega/2
        for i in range(N):
            p = out.s
            #calculate impedance
            zline1 = self.getz(p[:, 0, 0], f, z0)[0]
            zline2 = self.getz(p[:, 1, 1], f, z0)[0]
            #this is the transmission line to be removed
            TL1 = self.makeTL(zline1, z0, betal, 1)
            TL2 = self.makeTL(zline2, z0, betal, 1)
            sTL1 = nin.copy()
            sTL1.s = TL1
            sTL2 = nin.copy()
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

    def makeErrorBox_v7(self, s_dut, s2x, gamma, z0, pullback):
        f = s2x.frequency.f
        n = len(f)
        s212x = s2x.s[:, 1, 0]
        DC21 = self.dc_interp(s212x, f)
        x = np.argmax(irfft(concatenate(([DC21], s212x))))
        #define relative length
        #python first index is 0, thus 1 should be added to get the length
        l = 1. / (2 * (x + 1))
        #define the reflections to be mimicked
        s11dut = s_dut.s[:, 0, 0]
        s22dut = s_dut.s[:, 1, 1]
        #peel the fixture away and create the fixture model
        #python range to n-1, thus 1 to be added to have proper iteration number
        for i in range(x + 1 - pullback):
            zline1 = self.getz(s11dut, f, z0)[0]
            zline2 = self.getz(s22dut, f, z0)[0]
            TL1 = self.makeTL(zline1,z0,gamma,l)
            TL2 = self.makeTL(zline2,z0,gamma,l)
            sTL1 = s_dut.copy()
            sTL1.s = TL1
            sTL2 = s_dut.copy()
            sTL2.s = TL2
            if i == 0:
                errorbox1 = sTL1
                errorbox2 = sTL2
                _, z1 = self.getz(s11dut, f, z0)
                _, z2 = self.getz(s22dut, f, z0)
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
            #     abcd_in[j, :, :] = np.linalg.lstsq(abcd_TL1[j, :, :].T, np.linalg.lstsq(abcd_TL1[j, :, :], abcd_in[j, :, :], rcond=None)[0].T, rcond=None)[0].T
            # s_dut.a = abcd_in
            s11dut = s_dut.s[:, 0, 0]
            s22dut = s_dut.s[:, 1, 1]
            if self.verbose:
                if i == 0:
                    fig, axs = subplots(2, 2)
                    axs[0, 0].plot(z1, color = 'k')
                    axs[0, 0].set_xlim((0, x))
                    axs[0, 1].plot(z2, color = 'k')
                    axs[0, 1].set_xlim((0, x))
                    axs[1, 0].set_xlim((n-100, n+x*2+10))
                    axs[1, 1].set_xlim((n-100, n+x*2+10))
                _, zeb1 = self.getz(errorbox1.s[:, 0, 0], f, z0)
                _, zeb2 = self.getz(errorbox2.s[:, 0, 0], f, z0)
                _, zdut1 = self.getz(s11dut, f, z0)
                _, zdut2 = self.getz(s22dut, f, z0)
                axs[0, 0].plot(zeb1)
                axs[0, 1].plot(zeb2)
                axs[1, 0].plot(ifftshift(zdut1))
                axs[1, 1].plot(ifftshift(zdut2))
        return errorbox1, errorbox2.flipped()

    def makeErrorBox_v8(self, s_dut, s2x, gamma, z0, pullback):
        f = s2x.frequency.f
        n = len(f)
        s212x = s2x.s[:, 1, 0]
        # extract midpoint of 2x-thru
        DC21 = self.dc_interp(s212x, f)
        x = np.argmax(irfft(concatenate(([DC21], s212x))))
        #define relative length
        #python first index is 0, thus 1 should be added to get the length
        l = 1. / (2 * (x + 1))
        #define the reflections to be mimicked
        s11dut = s_dut.s[:, 0, 0]
        #peel the fixture away and create the fixture model
        #python range to n-1, thus 1 to be added to have proper iteration number
        for i in range(x + 1 - pullback):
            zline1 = self.getz(s11dut, f, z0)[0]
            TL1 = self.makeTL(zline1,z0,gamma,l)
            sTL1 = s_dut.copy()
            sTL1.s = TL1
            if i == 0:
                errorbox1 = sTL1
                _, z1 = self.getz(s11dut, f, z0)
            else:
                errorbox1 = errorbox1 ** sTL1
            # equivalent to function removeTL_side1(in,TL,z0)
            s_dut = sTL1.inv ** s_dut
            s11dut = s_dut.s[:, 0, 0]
            if self.verbose:
                if i == 0:
                    fig, axs = subplots(1, 2)
                    axs[0].plot(z1, color = 'k')
                    axs[0].set_xlim((0, x))
                    axs[1].set_xlim((n-100, n+x*2+10))
                _, zeb1 = self.getz(errorbox1.s[:, 0, 0], f, z0)
                _, zdut1 = self.getz(s11dut, f, z0)
                axs[0].plot(zeb1)
                axs[1].plot(ifftshift(zdut1))
        return errorbox1


    def split2xthru(self, s2xthru, sfix_dut_fix):

        f = sfix_dut_fix.frequency.f
        s = sfix_dut_fix.s

        # check for bad inputs
        # check for DC point
        if(f[0] == 0):
            warnings.warn(
                "DC point detected. The included DC point will not be used during extraction.",
                RuntimeWarning
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
               RuntimeWarning
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
               RuntimeWarning
               )

        # enforce Nyquist rate point
        if self.NRP_enable:
            sfix_dut_fix, TD = self.NRP(sfix_dut_fix)
            s2xthru, _ = self.NRP(s2xthru, -TD)

        # remove lead-in points
        if self.leadin > 0:
            _, temp1, temp2 = self.peelNPointsLossless(self.shiftNPoints(sfix_dut_fix, self.leadin), self.leadin)
            leadin1 = self.shiftOnePort(temp1, -self.leadin, 0)
            leadin2 = self.shiftOnePort(temp2, -self.leadin, 1)

        # calculate gamma
        #grabbing s21
        s212x = s2xthru.s[:, 1, 0]
        #get the attenuation and phase constant per length
        beta_per_length = -np.unwrap(np.angle(s212x))
        attenuation = np.abs(s2xthru.s[:,1,0])**2 / (1. - np.abs(s2xthru.s[:,0,0])**2)
        alpha_per_length = (10.0 * np.log10(attenuation)) / -8.686
        if self.bandwidth_limit == 0:
            #divide by 2*n + 1 to get prop constant per discrete unit length
            gamma = alpha_per_length + 1j * beta_per_length # gamma without DC
        else:
            #fit the attenuation up to the limited bandwidth
            bwl_x = np.argmin(np.abs(f - self.bandwidth_limit))
            X = np.array([np.sqrt(f[0:bwl_x+1]), f[0:bwl_x+1], f[0:bwl_x+1]**2])
            b = np.linalg.lstsq(X.conj().T, alpha_per_length[0:bwl_x+1], rcond=None)[0]
            alpha_per_length_fit = b[0] * np.sqrt(f) + b[1] * f + b[2] * f**2
            #divide by 2*n + 1 to get prop constant per discrete unit length
            gamma = alpha_per_length_fit + 1j * beta_per_length # gamma without DC

        # extract error boxes
        # make the both error box
        s_side1 = self.thru(sfix_dut_fix)
        s_side2 = self.thru(sfix_dut_fix)

        if self.pullback1 == self.pullback2 and self.side1 and self.side2:
            (s_side1, s_side2) = self.makeErrorBox_v7(sfix_dut_fix, s2xthru,
                                              gamma, self.z0, self.pullback1)
        elif self.side1 and self.side2:
            s_side1 = self.makeErrorBox_v8(sfix_dut_fix, s2xthru,
                                              gamma, self.z0, self.pullback1)
            s_side2 = self.makeErrorBox_v8(sfix_dut_fix.flipped(),s2xthru,
                                              gamma, self.z0, self.pullback2)
            # no need to flip FIX-2 as per IEEEP370 numbering recommandation
        elif self.side1:
            s_side1 = self.makeErrorBox_v8(sfix_dut_fix, s2xthru,
                                              gamma, self.z0, self.pullback1)
        elif self.side2:
            s_side2 = self.makeErrorBox_v8(sfix_dut_fix.flipped(),s2xthru,
                                              gamma, self.z0, self.pullback2)
        else:
            warnings.warn(
               "no output because no output was requested",
               RuntimeWarning
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
            s_side1 = self.add_dc(s_side1)
            s_side2 = self.add_dc(s_side2)

        # remove lead in
        if self.leadin > 0:
            s_side1 = leadin1 ** s_side1
            s_side2 = s_side2 ** leadin2

        # if Nyquist Rate Point enforcement is enabled
        if self.NRP_enable:
            s_side1, _ = self.NRP(s_side1, TD, 0)
            s_side2, _ = self.NRP(s_side2, TD, 1)

        return (s_side1, s_side2)


class IEEEP370_MM_ZC_2xThru(Deembedding):
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
    def __init__(self, dummy_2xthru, dummy_fix_dut_fix, name=None,
                 z0 = 50, port_order: str = 'second',
                 bandwidth_limit = 0,
                 pullback1 = 0, pullback2 = 0,
                 side1 = True, side2 = True,
                 NRP_enable = True, leadin = 1,
                 verbose = False,
                 *args, **kwargs):
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

        Deembedding.__init__(self, dummies, name, *args, **kwargs)
        self.se_side1, self.se_side2 = self.split2xthru(self.s2xthru,
                                                        self.sfix_dut_fix)

    def deembed(self, ntwk):
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
            warnings.warn('Network frequencies dont match dummy frequencies, attempting overlap.', RuntimeWarning)
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

    def split2xthru(self, se_2xthru, se_fdf):
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

        #convert back to single-ended
        mm_side1 = concat_ports([dm_dd.s_side1, dm_cc.s_side1], port_order = 'first')
        se_side1 = mm_side1.copy()
        se_side1.gmm2se(p = 2)
        mm_side2 = concat_ports([dm_dd.s_side2, dm_cc.s_side2], port_order = 'first')
        se_side2 = mm_side2.copy()
        se_side2.gmm2se(p = 2)

        return (se_side1, se_side2)
