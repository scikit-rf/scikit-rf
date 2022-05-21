r"""
Qfactor (:mod:`skrf.qfactor`)
========================================
Module for fitting Quality (Q) factor(s) from S-parameters.

Measurements of the Quality factor (Q-factor) is straightforward,
but to obtain uncertainty below 1%
(which is considered to be low for Q-factor measurement) requires attention
to several aspects of the experimental procedure.

This class implements methods for determining loaded and unloaded Q-factor
from frequency-domain S-parameters, that can be applied to measurements
of transmission or reflection.

Documentation and implementation are adapted from [#]_

Q-factor
--------
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    Qfactor

.. currentmodule:: skrf.qfactor


Loaded and Unloaded Q-factor
----------------------------
The Quality factor (Q-factor) of a resonator is defined by:

.. math::

    Q = \frac{2 \pi U}{\Delta U}


where :math:`U` is the average energy stored by the resonator and
:math:`\Delta U` is the decrease in the average stored energy per wave cycle
at the resonant frequency [#]_.

The loaded Q-factor, :math:`Q_L`, describes energy dissipation within the
entire resonant system comprising of the resonator itself and the instrument
used for observing resonances. The term loading refers to the effect that the
external circuit has on measured quantities.

The external circuit consists of the measuring instrument and uncalibrated lines,
but not the couplings of microwave resonators. Loading by an instrument that
has 50 Ohm impedance, such as a VNA, causes :math:`Q_L` to be reduced substantially
if strong coupling is used.

For most applications the quantity that is desired is the unloaded Q-factor :math:`Q_0`,
which is determined by energy dissipation associated with the resonator only
and therefore gives the best description of the resonant mode.

In other words, :math:`Q_0` is the Q-factor of the uncoupled resonator. The value of
:math:`Q_0` can be estimated from measurements of :math:`Q_L`, but cannot be measured directly.
:math:`Q_0` is largely governed by ohmic loss arising from surface currents
in the metal conductors (walls and loop couplings), and from dielectric loss
in any insulating materials that may be present.



Energy dissipation in the external circuit is characterised by the external Q-factor,
:math:`Q_e`. For both series and parallel equivalent circuits, the three
Q-factors are related by:

.. math::

    \frac{1}{Q_L} = \frac{1}{Q_0} + \frac{1}{Q_e}


The coupling factor :math:`\beta` is defined by:

.. math::

    \beta = \frac{Q_0}{Q_e}

where a a coupling factor is defined for each port.

The diameter of the Q-circle displayed by the VNA provides a visual
indication of whether the coupling is strong or weak. :math:`Q_0` can be
calculated from the measured :math:`Q_L` when coupling factor(s) are known.
Hence, coupling factors can be calculated from calibrated Q-circle diameters.

This class offers methods to
determine the uncoupled (unloaded) Q-factor :math:`Q_0` from advanced fittings.

Q-factor determination through equivalent-circuit models
--------------------------------------------------------
Characterisation of resonances from measurements in the frequency-domain
can be achieved through equivalent-circuit models. A high Q-factor resonator
(in practice, :math:`Q_L` > 100), the S-parameter response of a resonator
measured in a calibrated system with reference planes at the resonator couplings is:

.. math::

    S = S_D + d \frac{e^{−2j\delta}}{1 + j Q_L t}


where :math:`S_D` is the detuned S-parameter measured at frequencies far above or below
resonance, :math:`d` is The diameter of the Q-circle, :math:``delta` is a
real-valued constant that defines the orientation of the Q-circle, and :math:`t`
is the fractional offset frequency given by:

.. math::
    t = 2 \frac{f − f_L}{f_0} \approx 2 \frac{f − f_L}{f_L}


where :math:`f_L` is the loaded resonant frequency, :math:`f_0` The unloaded
resonant frequency and :math:`f` the frequency at which S is measured.
This equation can be applied to measurements by transmission (S21 or S12) or
reflection (S11 or S22).




References
----------
.. [#] "Q-factor Measurement by using a Vector Network Analyser",
    A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
    https://eprintspublications.npl.co.uk/9304/

.. [#] D. M. Pozar, Microwave engineering, 4th ed. J. Wiley, 2012.

.. [#] M. Sucher and J. Fox. Handbook of microwave measurements; Vol. 2, 3rd Ed. New York: Polytechnic Press, 1963.

.. [#] B. A. Galwas, ‘Scattering Matrix Description of Microwave Resonators’, IEEE Trans. Microwave Theory Techn., vol. 31, no. 8, pp. 669–671, Aug. 1983, doi: 10.1109/TMTT.1983.1131566.



"""
import numpy as np
from .network import Network, a2s
from .media import media
from .constants import INF, NumberLike
from typing import List, TYPE_CHECKING, Tuple, Union

# Available resonance types
RESONANCE_TYPES = ['reflection', 'reflection_method2',
                   'transmission', 'absorption']



class OptimizedResult(dict):
    """ Represent Q-factor optimisation result.

    Attributes
    ----------
    Q_L : float
        Loaded Quality factor
    f_L : float
        Resonance frequency [Hz]
    success: bool
        Is the fit method has been successfully performed
    method: str
        Fitting method used.
    m1, m2, m3, ...: float
        Coefficients described in [#]_
    number_iterations: int
        Number of iterations performed.

    Notes
    -----
    `OptimizedResult` may have additional attributes not listed here depending
    on the specific fitting method used. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizedResult.keys` method.

    References
    ----------
    .. [#] "Q-factor Measurement by using a Vector Network Analyser",
        A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
        https://eprintspublications.npl.co.uk/9304/

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class Qfactor(object):
    """
    Q-factor calculation.

    Parameters
    ----------
    ntwk : :class:`~skrf.network.Network` object
        scikit-rf Network
    res_type : str
        Specifies the resonance type: 'reflection', 'transmission',
        'reflection_method2' or 'absorption'.
        'reflection' is generally suited for undercoupled resonators,
        while 'reflection_method2' is favoured for coupling with large loop. 
    Q_L0 : float, optional. Default is None.
        Estimated loaded Q-factor, used to improve fitting.
    f_L0 : float, optional. Default is None.
        Estimated loaded resonant frequency, used to improve fitting [Hz]
        If None, automatically search for the min or max,
        depending on the resonance type defined by `res_type`.
    verbose : bool, optional. Default is False.
        Boolean flag controlling output of information to the console.

    Notes
    -----
    Uncalibrated line should be de-embedded (if it has significant
    length) from the S-parameter data before calling the functions
    in this module to get best results.

    """

    def __init__(self,
                 ntwk: Network,
                 res_type: str,
                 Q_L0: Union[None, float] = None,
                 f_L0: Union[None, float] = None,
                 verbose: bool = False):
        """Q-factor initializer."""
        # check ntwk is a 1-port
        if ntwk.nports != 1:
            raise ValueError('The Network is not a 1-port Network.')
        if res_type not in RESONANCE_TYPES:
            raise ValueError(f'res_type must be in: {RESONANCE_TYPES}.')

        self.s = ntwk.s
        self.f = ntwk.f
        self.f_scaled = ntwk.frequency.f_scaled
        self._ntwk = ntwk
        self.res_type = res_type
        self.tol = 1.0e-5
        self.verbose = verbose

        self.N = len(self.f)

        # step 1: initial_fit. Deduce premilinary values for Q_L and f_L.
        self._initial_fit(self.N, Q_L0, f_L0)


    def fit(self, method: str="NLQFIT6", loop_plan: str = 'fwfwc'):
        """Fit Q-factor from S-parameter data.

        Fitting overwrites the parameters `Q_L` and `f_L`.

        Parameters
        ----------
        method : str, optional
            Fitting method : 'NLQFIT6' (default), 'NLQFIT7', 'NLQFIT8':
            'NLQFIT6': Least Square Minimum of Eq.21 [#]_ with 6 coeffcients.
            'NLQFIT7': Least Square Minimum of Eq.26 [#]_ with 7 coeffcients,
                       including one that characterize the trans. line length.
            'NLQFIT8': Least Square Minimum of Eq.43 [#]_ with 8 coeffcients,
                       A model for frequency-dependent leakage.
        loop_plan : str, optional
            Defines order of steps used by the fitting process.
            The convergence algorithm uses a number of steps set by loop_plan,
            a string of characters as follows:
              f - fit once without testing for convergence
              c - repeated fit, iterating until convergence is obtained
              w - re-calculate weighting factors on basis of previous fit
                  Initially the weighting factors are all unity.
              The first character in loop_plan must not be w.
            e.g.: 'fwfwc' (default).

        Returns
        -------
        optimized_results : OptimisedResult

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/

        """

        for op in loop_plan:
            if op not in ['f', 'c', 'w']:
                raise ValueError("Unexpected character in loop_plan")
        if loop_plan[-1] == "w":
            raise ValueError("Last item in loop_plan must not be w (weight calculation)")
        if loop_plan[0] == "w":
            raise ValueError("First item in loop_plan must not be w (weight calculation)")
        if loop_plan[-1] != "c":
            raise Warning("Warning: Last item in loop_plan is not c so convergence not tested!")

        self.method = method
        self.loop_plan = loop_plan

        # step 2: least square fitting
        if method == 'NLQFIT6':
            res = self._optimise_fit6(self.N)
        elif method == 'NLQFIT7':
            res = self._optimise_fit7(self.N)
        elif method == 'NLQFIT8':
            res = self._optimise_fit8(self.N)

        # overwrite results
        self.Q_L = res.Q_L
        self.f_L = res.f_L

        return res

    @staticmethod
    def angular_weights(f, f_L, Q_L):
        r"""Diagonal elements W_i of weights matrix.

        .. math::

            W_i = \frac{1}{\left[ \frac{2 Q_L (f_i - f_L)}{f_L} \right]^2 + 1}


        The weights are needed when frequencies are equally-spaced
        (rather than points equally spaced around the Q-circle), and help
        reducing systematic error [#]_.

        Parameters
        ----------
        f : np.ndarray
            Frequency values array.
        f_L : float
            Loaded resonant frequency.
        Q_L : float
            Loaded Q-factor.

        Returns
        -------
        W_i : np.ndarray
            Weighting factors in proportion to the rate of change of angle
            with frequency relative to the centre of the Q-circle.

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/
            section 2.4, eqn. (28).

        """
        ptmp = 2 * Q_L * (f - f_L) / f_L
        W_i = 1 / (ptmp ** 2 + 1)
        return W_i

    def _initial_fit(self,
                    N: int,
                    Q_L0: Union[float, None] = None,
                    f_L0: Union[None, float] = None
                    ):
        """Initial Linear least squares Q-factor fit.

        As this is not optimised in this function (use `fit`), the solution
        will only be approximate. This method is called during the
        initialization of the `Qfactor` class. Note that a reasonable estimate
        for the resonant frequency should be supplied is multiple resonances
        are present.

        Also calculate the internal parameters a, b and QL from ([#]_, eqn. 17)

        Parameters
        ----------
        N : int
            Number of points.
        Q_L0 : float, optional. Default is None.
            Estimated loaded Q-factor (will be improved by fitting).
        f_L0 : float, optional. Default is None.
            Estimated loaded resonant frequency, used to improve fitting [Hz]

        Returns
        -------
        None

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/
            section 2.1, eqn. (17).

        """
        if f_L0 is None:
            # search for the initial value of the resonance frequency
            if self.res_type in ['reflection', 'reflection_method2', 'absorption']:
                # Find minimum in |S11|
                index_min = np.argmin(np.abs(self.s))
                f_L0 = self.f[index_min]
            else:
                # Find peak in |S21|
                index_max = np.argmax(np.abs(self.s))
                f_L0 = self.f[index_max]

        # Q_L0 : An order-of-magnitude estimate for Q_L-factor
        if Q_L0 is None:
            Tol = self.tol * np.argmax(np.abs(self.s))
            # The value 5.0 should work well
            # for initial and optimised fits (Section 2.6).
            mult = 5.0
            Q_L0 = mult * f_L0/(self.f[-1] - self.f[0])

        if self.verbose:
            print(f'Initial estimation: Q_L0={Q_L0}, f_L0={f_L0}')

        N2 = 2 * N
        M = np.zeros([N2, 5])
        G = np.zeros(N2)[:, np.newaxis]

        for i in range(N):
            i2 = i + N
            t = 2.0 * (self.f[i] / f_L0 - 1.0)
            y = 1.0 / complex(1.0, Q_L0 * t)
            v = t * y
            v1 = y * self.s[i]
            G[i] = v1.real
            G[i2] = v1.imag
            v2 = v1 * t
            M[i, :] = v.real, -v.imag, y.real, -y.imag, v2.imag
            M[i2, :] = v.imag, v.real, y.imag, y.real, -v2.real

        T = M.transpose()  # unweighted
        C = T @ M
        q = T @ G
        sv = np.linalg.solve(C, q)
        a_re, a_im, b_re, b_im, Q_L = sv

        self._a = a_re + 1j*a_im
        self._b = b_re + 1j*b_im
        self.Q_L = Q_L
        self.f_L = f_L0

        if self.verbose:
            print(f'Preliminary estimation: Q_L={self.Q_L}, f_L={self.f_L}')


    def _optimise_fit6(self, N: int):
        """Iterative non-linear fit, NLQFIT6 Step (2).

        Optimised fit of Q-factor (Q_L) and resonant frequency (f_L)
        by the gradient-descent method [#]_.

        Uses the results of the initial fit as the starting
        values for the iteration.

        Parameters
        ----------
        N : int
            Number of points.

        Returns
        -------
        res : OptimizedResult
            Fitted values.

        References
        ----------
        .. [#] MAT 58, section 2.2, eqn. (22).

        """
        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = self._a.imag / self.Q_L  # a''/QL
        m2 = -self._a.real / self.Q_L
        m3 = self._b.real - m1
        m4 = self._b.imag - m2
        m5 = self.Q_L
        Flwst = self.f[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * m5 / self.f_L
        last_op = "n"
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in self.loop_plan:
            if op == "w":
                PV = self.angular_weights(self.f, Flwst * float(m5) / float(m6), float(m5))
                # PV = self.angular_weights(m5)
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if self.verbose:
                    print("Op w, Calculate weights")
                last_op = "n"
            elif op == "c":
                seek_convergence = True
            elif op == "f":
                seek_convergence = False

            TerminationConditionMet = False
            RMS_Error = None
            while not (TerminationConditionMet):
                number_iterations += 1
                M = np.zeros([N2, 6])  # X is the transpose of M
                G = np.zeros(N2)[:, np.newaxis]
                c1 = complex(-m4, m3)
                c2 = complex(m1, m2)
                c3 = complex(m3, m4)
                for i in range(N):
                    i2 = i + N
                    y = 1.0 / complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    u = c1 * y * y * 2
                    u2 = -u * self.f[i] / Flwst
                    M[i, :] = 1.0, 0.0, y.real, -y.imag, u.real, u2.real
                    M[i2, :] = 0.0, 1.0, y.imag, y.real, u.imag, u2.imag
                    v = c2 + c3 * y
                    r = self.s[i] - v  # residual
                    G[i] = r.real
                    G[i2] = r.imag
                X = M.transpose()
                T = np.multiply(X, PV2)
                C = np.dot(T, M)
                q = np.dot(T, G)
                dm = np.linalg.solve(C, q)
                m1 += dm[0]
                m2 += dm[1]
                m3 += dm[2]
                m4 += dm[3]
                m5 += dm[4]
                m6 += dm[5]
                del G, X, T, C, dm
                iterations = iterations + 1
                if RMS_Error is not None:
                    Last_RMS_Error = RMS_Error
                else:
                    Last_RMS_Error = None
                SumNum = 0.0
                SumDen = 0.0
                for i in range(N):
                    den = complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    ip = PV[i]
                    E = self.s[i] - complex(m1, m2) - complex(m3, m4) / den
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if self.verbose:
                    if last_op == "c":
                        print(f"Iteration {iterations}, RMS Error: {RMS_Error}")
                    else:
                        print(f"op {op}, Iteration {iterations}, RMS Error: {RMS_Error}")
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < self.tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if self.verbose:
                print("Optimization done.")


        return OptimizedResult({
            'success': TerminationConditionMet,
            'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
            'Q_L': m5,
            'f_L': m5 * Flwst / m6,
            'weighting_ratio': weighting_ratio,
            'number_iterations': number_iterations,
            'RMS_Error': RMS_Error,
            'method': self.method,
            })

    def _optimise_fit7(self, N):
        """Iterative non-linear fit, NLQFIT7 Step (2).

        Optimised fit of Q-factor (QL) and resonant frequency (FL)
        by the gradient-descent method.

        Uses the results of the initial fit (sv) as the starting
        values for the iteration.

        Parameters
        ----------
        F : np.ndarray
            Frequency points
        S : np.ndarray
            Complex data S-parameter to be fitted.
        N : int
            Number of points.
        Fseed : float
            Estimated resonant frequency.
        sv : list
            Initial solution (numpy vector or a list) found with initial_fit.
        loop_plan : str
            Characters which defines order of steps used by the fitting process
            e.g. 'fwfwc':
                'f' - fit once without testing for convergence.
                'c' - repeated fit, iterating until convergence is obtained.
                'w' - re-calculate weighting factors on basis of previous fit.
        Tol : float
            Criterion for the convergence test.
                    Recommend using 1.0E-5 for reflection or max(abs(S))*1.0E-5
                    for transmission.
        quiet : bool
            Boolean flag controlling output of information to the console.

        Returns
        -------
        list
            list of fitted parameters: [m1, m2, m3, m4, m5, m5 * Flwst / m6, m7 / Flwst]
        weighting_ratio : float
        number_iterations : int
        RMS_Error : float

        References
        ----------
        .. [#] MAT 58, section 2.3, eqn. (26).

        """
        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = self._a.imag / self.Q_L  # a''/QL
        m2 = -self._a.real / self.Q_L
        m3 = self._b.real - m1
        m4 = self._b.imag - m2
        m5 = self.Q_L
        Flwst = self.f[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * self.Q_L / self.f_L
        m7 = 0.0
        last_op = "n"
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in self.loop_plan:

            if op == "w":
                PV = self.angular_weights(self.f, Flwst * m5 / m6, m5)
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if self.verbose:
                    print("Op w, Calculate weights")
                last_op = "n"
                continue
            if op == "c":
                seek_convergence = True
            elif op == "f":
                seek_convergence = False
            else:
                assert 0, "Unexpected character in loop_plan"

            TerminationConditionMet = False
            RMS_Error = None
            while not (TerminationConditionMet):
                number_iterations += 1
                M = np.zeros([N2, 7])
                G = np.zeros(N2)[:, np.newaxis]
                c1 = complex(-m4, m3)
                c2 = complex(m1, m2)
                c3 = complex(m3, m4)
                for i in range(N):
                    i2 = i + N
                    y = 1.0 / complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    fdn = self.f[i] / Flwst - m5 / m6
                    pj = complex(0.0, m7 * fdn)
                    expm7 = np.exp(pj)
                    ym = y * expm7
                    u = c1 * y * ym * 2
                    u2 = -u * self.f[i] / Flwst
                    v = (c2 + y * c3) * expm7
                    u3 = v * fdn
                    M[i, :] = (
                        expm7.real,
                        -expm7.imag,
                        ym.real,
                        -ym.imag,
                        u.real,
                        u2.real,
                        -u3.imag,
                    )
                    M[i2, :] = (
                        expm7.imag,
                        expm7.real,
                        ym.imag,
                        ym.real,
                        u.imag,
                        u2.imag,
                        u3.real,
                    )
                    r = self.s[i] - v  # residual
                    G[i] = r.real
                    G[i2] = r.imag
                X = M.transpose()
                T = np.multiply(X, PV2)
                C = np.dot(T, M)
                q = np.dot(T, G)
                dm = np.linalg.solve(C, q)
                m1 += dm[0]
                m2 += dm[1]
                m3 += dm[2]
                m4 += dm[3]
                m5 += dm[4]
                m6 += dm[5]
                m7 += dm[6]
                del G, X, T, C, dm
                iterations = iterations + 1
                if RMS_Error is not None:
                    Last_RMS_Error = RMS_Error
                else:
                    Last_RMS_Error = None

                SumNum = 0.0
                SumDen = 0.0
                for i in range(N):
                    fdn = self.f[i] / Flwst - m5 / m6
                    den = complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    pj = complex(0.0, m7 * fdn)
                    E = self.s[i] - (c2 + c3 / den) * np.exp(pj)
                    ip = PV[i]
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if self.verbose:
                    if last_op == "c":
                        print(f"Iteration {iterations}, RMS Error: {RMS_Error}")
                    else:
                        print(f"op {op}, Iteration {iterations}, RMS Error: {RMS_Error}")
 
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < self.tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if self.verbose:
                print("Optimization done.")

        return OptimizedResult({
            'success': TerminationConditionMet,
            'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
            'Q_L': m5,
            'f_L': m5 * Flwst / m6,
            'm7a' : m7 / Flwst,
            'weighting_ratio': weighting_ratio,
            'number_iterations': number_iterations,
            'RMS_Error': RMS_Error,
            'method': self.method,
            })

    def _optimise_fit8(self, N):
        """Iterative non-linear fit, NLQFIT8 Step (2).

        Optimised fit of Q-factor (QL) and resonant frequency (FL)
        by the gradient-descent method.

        Uses the results of the initial fit (sv) as the starting
        values for the iteration.

        Parameters
        ----------
        F : np.ndarray
            Frequency points
        S : np.ndarray
            Complex data S-parameter to be fitted.
        N : int
            Number of points.
        Fseed : float
            Estimated resonant frequency.
        sv : list
            Initial solution (numpy vector or a list) found with initial_fit.
        loop_plan : str
            Characters which defines order of steps used by the fitting process
            e.g. 'fwfwc':
                'f' - fit once without testing for convergence.
                'c' - repeated fit, iterating until convergence is obtained.
                'w' - re-calculate weighting factors on basis of previous fit.
        Tol : float
            Criterion for the convergence test.
                    Recommend using 1.0E-5 for reflection or max(abs(S))*1.0E-5
                    for transmission.
        quiet : bool
            Boolean flag controlling output of information to the console.

        Returns
        -------
        list
            list of fitted parameters: [m1, m2, m3, m4, m5, m5 * Flwst / m6, m7 / Flwst]
        weighting_ratio : float
        number_iterations : int
        RMS_Error : float

        References
        ----------
        .. [#] MAT 58, sec 4.5, eqn. (43).

        """
        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = self._a.imag / self.Q_L  # a''/QL
        m2 = -self._a.real / self.Q_L
        m3 = self._b.real - m1
        m4 = self._b.imag - m2
        m5 = self.Q_L
        Flwst = self.f[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * self.Q_L / self.f_L
        m8 = 0.0
        m9 = 0.0
        last_op = "n"
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in self.loop_plan:

            if op == "w":  # Fr                       QL
                PV = self.angular_weights(self.f, Flwst * float(m5) / float(m6), float(m5))
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if self.verbose:
                    print("Op w, Calculate weights")
                last_op = "n"
                continue
            if op == "c":
                seek_convergence = True
            elif op == "f":
                seek_convergence = False
            else:
                assert 0, "Unexpected character in loop_plan"

            TerminationConditionMet = False
            RMS_Error = None
            while not (TerminationConditionMet):
                number_iterations += 1
                M = np.zeros([N2, 8])
                G = np.zeros(N2)[:, np.newaxis]
                c1 = complex(-m4, m3)
                c2 = complex(m1, m2)
                c3 = complex(m3, m4)
                for i in range(N):
                    i2 = i + N
                    y = 1.0 / complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    u = c1 * y * y * 2
                    u2 = -u * self.f[i] / Flwst
                    FL = Flwst * m5 / m6
                    t = 2 * (self.f[i] - FL) / FL
                    M[i, :] = 1.0, 0.0, y.real, -y.imag, u.real, u2.real, t, 0.0
                    M[i2, :] = 0.0, 1.0, y.imag, y.real, u.imag, u2.imag, 0.0, t
                    v = c2 + c3 * y + (m8 + 1j * m9) * t
                    r = self.s[i] - v  # residual
                    G[i] = r.real
                    G[i2] = r.imag
                X = M.transpose()
                T = np.multiply(X, PV2)
                C = np.dot(T, M)
                q = np.dot(T, G)
                dm = np.linalg.solve(C, q)
                m1 += dm[0]
                m2 += dm[1]
                m3 += dm[2]
                m4 += dm[3]
                m5 += dm[4]
                m6 += dm[5]
                m8 += dm[6]
                m9 += dm[7]
                del G, X, T, C, dm
                iterations = iterations + 1
                if RMS_Error is not None:
                    Last_RMS_Error = RMS_Error
                else:
                    Last_RMS_Error = None

                SumNum = 0.0
                SumDen = 0.0
                for i in range(N):
                    den = complex(1.0, 2 * (m6 * self.f[i] / Flwst - m5))
                    FL = Flwst * m5 / m6
                    t = 2 * (self.f[i] - FL) / FL
                    ip = PV[i]
                    E = (
                        self.s[i]
                        - complex(m1, m2)
                        - complex(m8, m9) * t
                        - complex(m3, m4) / den
                    )
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if self.verbose:
                    if last_op == "c":
                        print(f"Iteration {iterations}, RMS Error: {RMS_Error}")
                    else:
                        print(f"{op}, Iteration {iterations}, RMS Error: {RMS_Error}")
 
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < self.tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if self.verbose:
                print("Optimization done.")

        return OptimizedResult({
            'success': TerminationConditionMet,
            'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
            'Q_L': m5,
            'f_L': m5 * Flwst / m6,
            'weighting_ratio': weighting_ratio,
            'number_iterations': number_iterations,
            'RMS_Error': RMS_Error,
            'method': self.method,
            })

    def Q_circle(self, opt_res: OptimizedResult, A: Union[str, float]) -> list:
        """Q-circle diameter.

        Defined as [#]_:

        .. math::

            d = A|b + j a/Q_L|

        Parameters
        ----------
        opt_res : OptimizedResult
            solution produced by the `fit` method.
        A : float or str
            Scaling factor as defined in MAT 58 [#]_.
            For `reflection` resonance type, can be set as 'AUTO'
            to use the magnitude of the fitted detuned reflection coefficient (gamma_V)

        Returns
        -------
        diam : float
            Q-circle diameter d.
        S_V : complex
            Off-resonance Reflection Coefficient.
        S_T : complex
            Tuned Reflection Coefficient.

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/,
            section 2.5, eqn. (31).

        """
        # m1 : real part of cal_gamma_V
        # m2 : imag part of cal_gamma_V
        # m3 : real part of b + j a/Q_L
        # m4 : imag part of b + j a/Q_L
        m1, m2, m3, m4 = [opt_res[key] for key in ['m1', 'm2', 'm3', 'm4']]

        if isinstance(A, str) and A.upper() == 'AUTO':
            A = 1.0 / abs(complex(m1, m2))  # scale to gammaV if 'AUTO'
        elif not isinstance(A, (int, float)):
            raise ValueError("A should be a float or 'AUTO'")

        aqratio = complex(m1, m2)
        b = complex(m1 + m3, m2 + m4)
        diam = abs(b - aqratio) * A
        S_V = complex(m1, m2) * A
        S_T = b * A
        return diam, S_V, S_T


    def Q_unloaded(self, opt_res: OptimizedResult, A: float) -> float:
        """Unloaded Q-factor Q0.

        The value of the unloaded Q-factor Q0 cannot be measured directly but
        can be estimated from the measurement of the loaded Q-factor Q_L.

        Parameters
        ----------
        opt_res : OptimizedResult
            solution produced by the `fit` method.
        A : float or str
            Scaling factor as defined in MAT 58 [#]_.
            For `reflection` resonance type, can be set as 'AUTO'
            to use the magnitude of the fitted detuned reflection coefficient (gamma_V)

        Returns
        -------
        Q0 : float
            Unloaded Q-factor.

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/
        .. [#] "The Physics of Superconducting Microwave Resonators"
            Gao, Jiansong (2008), doi:10.7907/RAT0-VM75.
            https://resolver.caltech.edu/CaltechETD:etd-06092008-235549

        """
        if isinstance(A, str) and A.upper() == "AUTO":
            auto_flag = True
        elif isinstance(A, (int, float)):
            auto_flag = False
        else:
            raise ValueError("Illegal Scaling factor; should be a float or 'AUTO'")

        m1, m2, m3, m4, m5 = [opt_res[key] for key in ['m1', 'm2', 'm3', 'm4', 'Q_L']]
        FL = opt_res['f_L']

        if self.res_type == "transmission":
            if auto_flag:
                raise ValueError('Scaling factor must not be "Auto" for transmission case')
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(opt_res, A)
            if cal_diam == 1.0:
                raise ZeroDivisionError("Divide by zero forestalled in calculation of Qo")
            Q0 = m5 / (1.0 - cal_diam)

        elif self.res_type == "reflection":
            if auto_flag:
                if self.verbose:
                    print('A is "Auto": using fitted data to estimate it')
                A = 1.0 / abs(complex(m1, m2))  # scale to gammaV if 'AUTO'
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(opt_res, A)
            cal_touching_circle_diam = 2.0
            if self.verbose:
                print(f"Q-circle diam = {cal_diam}, touching_circle_diam = {cal_touching_circle_diam}")
            den = cal_touching_circle_diam / cal_diam - 1.0
            Q0 = m5 * (1.0 + 1.0 / den)

        elif self.res_type == "reflection_method2":
            if auto_flag:
                raise ValueError('Scaling factor must not be "Auto" for Method 2')
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(opt_res, A)
            gv = abs(cal_gamma_V)
            gv2 = gv * gv
            mb = abs(cal_gamma_T)
            cosphi = (gv2 + cal_diam * cal_diam - mb * mb) / (
                2.0 * gv * cal_diam
            )  # Cosine rule
            cal_touching_circle_diam = (1.0 - gv2) / (1.0 - gv * cosphi)
            if self.verbose:
                print(f"Q-circle diam = {cal_diam}, touching_circle_diam = {cal_touching_circle_diam}")
            den = cal_touching_circle_diam / cal_diam - 1.0
            Q0 = m5 * (1.0 + 1.0 / den)

        elif self.res_type == "notch" or self.res_type == "absorption":  # By transmission
            if auto_flag:
                if self.verbose:
                    print(
                        'Notch/absorption Qo calculation: Supplied A is "Auto", so using fitted data to estimate it'
                    )
                # scale to gammaV if 'AUTO'
                A = 1.0 / abs(complex(m1, m2))
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(opt_res, A)
            if self.verbose:
                print(f"Q-circle diam = {cal_diam}")
            if cal_diam == 1.0:
                raise ZeroDivisionError("Divide by zero forestalled in calculation of Qo")
            den = 1.0 / cal_diam - 1.0  # Gao thesis (2008) 4.35 and 4.40
            Q0 = m5 * (
                1.0 + 1.0 / den
            )  # https://resolver.caltech.edu/CaltechETD:etd-06092008-235549
            # For this type of resonator, critical coupling occurs for cal_diam = 0.5.
        else:
            raise ValueError("Unknown resonance type {self.res_type}")

        return Q0

    @staticmethod
    def s_model(f, f_L, Q_L, d, delta, s_D):
        """
        S-parameter response of an equivalent circuit model resonator. 

        Characterisation of resonances from measurements in the frequency-domain
        can be achieved through equivalent-circuit models. For high Q-factor resonator
        (in practice, :math:`Q_L` > 100), the S-parameter response of a resonator
        measured in a calibrated system with reference planes at the resonator couplings is [#]_:
        
        .. math::
        
            S = S_D + d \frac{e^{−2j\delta}}{1 + j Q_L t}

        Parameters
        ----------
        f : np.ndarray
            frequency array [Hz]
        f_L : float
            Resonant frequency [Hz]
        Q_L : float
            Loaded Q-factor
        d : float
            Diameter of the Q-Circle
        delta : float
            constant that defines the orientation of the Q-Circle
        s_D : complex
            Detuned S-parameter, measured far above of below the resonant freq
            
        Returns
        -------
        s : np.ndarray
            S-parameter response.

        References
        ----------
        .. [#] "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/,
            section 1.1, eqn. (1).

        """
        # fractional offet frequency
        t = f/f_L - f_L/f

        s = s_D + d/(1 + 1j*Q_L*t)*np.exp(-1j*2*delta)

        return s
