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

"""

from abc import ABC, abstractmethod
from ..frequency import *
from ..network import *


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
                raise(ValueError('Dummy Networks dont have matching frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

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
    and `dummy_short`. When :func:`Deembedding.deembed` is applied, the 
    Y-parameters of the dummy_open are subtracted from the DUT measurement, 
    followed by subtraction of Z-parameters of dummy-short.

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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()

        # remove open parasitics
        caled.y = ntwk.y - self.open.y
        # remove short parasitics
        caled.z = caled.z - self.short.z

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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()

        # remove open parasitics
        caled.y = ntwk.y - self.open.y

        return caled


class ShortOpen(Deembedding):
    """
    Remove short parasitics followed by open parasitics.

    A deembedding object is created with two dummy measurements: `dummy_open` 
    and `dummy_short`. When :func:`Deembedding.deembed` is applied, the 
    Z-parameters of the dummy_short are subtracted from the DUT measurement, 
    followed by subtraction of Y-parameters of dummy_open.

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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()

        # remove short parasitics
        caled.z = ntwk.z - self.short.z
        # remove open parasitics
        caled.y = caled.y - self.open.y

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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()

        # remove short parasitics
        caled.z = ntwk.z - self.short.z

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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        left = self.thru.copy()
        left_y = left.y
        left_y[:,0,0] = (self.thru.y[:,0,0] - self.thru.y[:,1,0] + self.thru.y[:,1,1] - self.thru.y[:,0,1]) / 2
        left_y[:,0,1] = self.thru.y[:,1,0] + self.thru.y[:,0,1]
        left_y[:,1,0] = self.thru.y[:,1,0] + self.thru.y[:,0,1]
        left_y[:,1,1] = - self.thru.y[:,1,0] - self.thru.y[:,0,1]
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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        left = self.thru.copy()
        left_z = left.z
        left_z[:,0,0] = (self.thru.z[:,0,0] + self.thru.z[:,1,0] + self.thru.z[:,1,1] + self.thru.z[:,0,1]) / 2
        left_z[:,0,1] = self.thru.z[:,1,0] + self.thru.z[:,0,1]
        left_z[:,1,0] = self.thru.z[:,1,0] + self.thru.z[:,0,1]
        left_z[:,1,1] = self.thru.z[:,1,0] + self.thru.z[:,0,1]
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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()
        h = ntwk ** self.thru.inv
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
            raise(ValueError('Network frequencies dont match dummy frequencies.'))

        # TODO: attempt to interpolate if frequencies do not match

        caled = ntwk.copy()
        h = ntwk ** self.thru.inv
        h_ = h.flipped()
        caled.z = (h.z + h_.z) / 2

        return caled