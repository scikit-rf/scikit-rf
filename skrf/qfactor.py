# -*- coding: utf-8 -*-
r"""
Qfactor (:mod:`skrf.qfactor`)
========================================

Q-factor determination from Network-parameters fitting.

Measurements of Q-factor are straightforward, but to obtain uncertainty <1%
(which is considered to be low for Q-factor measurement) requires attention
to several aspects of the experimental procedure.

This class implements methods for determining Q-factor from frequency-domain
S-parameters, that can be applied to measurements of transmission or reflection.

Documenation and implementation are adapted from [#]_

Q-factor
--------
.. autosummary::
   :toctree: generated/

Loaded and Unloaded Q-factor
----------------------------
The Q-factor of a resonator is defined by:

.. maths::

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
in any insulating materials that may be present. This class offers method to
determine the uncoupled (unloaded) Q-factor :math:`Q_0` from advanced fittings.

Q-factor determination through equivalent-circuit models
--------------------------------------------------------
Characterisation of resonances from measurements in the frequency-domain
can be achieved through equivalent-circuit models. A high Q-factor resonator
(in practice, :math:`Q_L` > 100), the S-parameter response of a resonator
measured in a calibrated system with reference planes at the resonator couplings is:

.. maths::

    S = S_D + d \frac{e^{−2j\delta}}{1 + j Q_L t}


where :math:`S_D` is the detuned S-parameter measured at frequencies far above or below
resonance, :math:`d` is The diameter of the Q-circle, :math:``delta` is a
real-valued constant that defines the orientation of the Q-circle, and :math:`t`
is the fractional offset frequency given by:

.. maths::
    t = 2 \frac{f − f_L}{f_0} \approx 2 \frac{f − f_L}{f_L}


where :math:`f_L` is the loaded resonant frequency, :math:`f_0` The unloaded
resonant frequency and :math:`f` the frequency at which S is measured.
This equation can be applied to measurements by transmission (S21 or S12) or
reflection (S11 or S22).

Time-domain (“ring down”) methods [#]_ enable measurement of the loaded
Q-factor :math:`Q_L`, in which :math:`U` and :math:`\Delta U` pertain to the
entire system comprising of the resonator and the instrument that is used for
observing resonances. The time-domain method requires excitation of a
resonance followed by a measurement of the exponential decay of the amplitude
(or stored energy).


References
----------
.. [#] "Q-factor Measurement by using a Vector Network Analyser",
    A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
    https://eprintspublications.npl.co.uk/9304/

.. [#] D. M. Pozar, Microwave engineering, 4th ed. J. Wiley, 2012.

.. [#] M. Sucher and J. Fox. Handbook of microwave measurements; Vol. 2, 3rd Ed. New York: Polytechnic Press, 1963.




"""
import numpy as np
from .network import Network, a2s
from .media import media
from .constants import INF, NumberLike
from typing import List, TYPE_CHECKING, Tuple


class Qfactor:
    """
    Q-factor calculation.



    Parameters
    ----------
    ntwk : :class:`~skrf.network.Network` object
        scikit-rf Network
    f_res : float, optional
        Estimated resonant frequency (not fitted) [Hz]
        Default is None (search for min/max)
    trmode : str
        specifies the resonance type :
            'transmission', 'reflection_method1', 'reflection_method2' or 'absorption'.
    method : str, optional
        Fitting method : 'NLQFIT6' (default), 'NLQFIT7', 'NLQFIT8'.
    loop_plan : str, optional
        Defines order of steps used by the fitting process.
        The convergence algorithm uses a number of steps set by loop_plan,
        a string of characters as follows:
          f - fit once without testing for convergence
          c - repeated fit, iterating until convergence is obtained
          w - re-calculate weighting factors on basis of previous fit
              Initially the weighting factors are all unity.
          The first character in loop_plan must not be w.
        e.g.: 'fwfwc'.
        Default is

    Notes
    -----
    Uncalibrated line should be de-embedded (if it has significant
    length) from the S-parameter data before calling the functions
    in this module.

    """

    def __init__(self, method="NLQFIT6"):
        """
        Q-factor initializer.
        """

    def angular_weights(self, f, f_res, Q_L):
        """
        Diagonal elements W_i of weights matrix.

        The weights are needed when frequencies are equally-spaced
        (rather than points equally spaced around the Q-circle), and help
        reducing systematic error.

        Parameters
        ----------
        Q_L : float
            Loaded Q-factor

        Returns
        -------
        W_i : np.ndarray
            Weighting factors in proportion to the rate of change of angle
            with frequency relative to the centre of the Q-circle.

        References
        ----------
        .. [#] MAT 58, section 2.4, eqn. (28).

        """
        ptmp = 2 * Q_L * (f - f_res) / f_res
        W_i = 1 / (np.abs(ptmp) ** 2 + 1)
        return W_i

    def initial_fit(self, f, s, N, f_res, Q_L0):
        """
        Initial Linear least squares Q-factor fit, step (1).

        A reasonable estimate for the resonant frequency must be
        supplied. As this is not optimised in this function, the
        solution will only be approximate.

        Parameters
        ----------
        s : np.ndarray
            Scattering parameters
        N : int
            Number of points.
        Q_L0 : float
            Estimated Q_L (will be improved by fitting).

        Returns
        -------
        sv : list
            sv = [a', a'', b', b'', QL]
            Output data (MAT58 eqn. 17):

        References
        ----------
        .. [#] MAT 58, section 2.1, eqn. (17).

        """
        N2 = 2 * N
        M = np.zeros([N2, 5])
        G = np.zeros(N2)[:, np.newaxis]

        for i in range(N):
            i2 = i + N
            t = 2.0 * (f[i] / f_res - 1.0)
            y = 1.0 / complex(1.0, Q_L0 * t)
            v = t * y
            v1 = y * s[i]
            G[i] = v1.real
            G[i2] = v1.imag
            v2 = v1 * t
            M[i, :] = v.real, -v.imag, y.real, -y.imag, v2.imag
            M[i2, :] = v.imag, v.real, y.imag, y.real, -v2.real
        # X = M.transpose()
        # P = [1.0]*N2 # Weights if required
        # T = np.multiply(X,P)
        T = M.transpose()  # unweighted
        C = np.dot(T, M)
        q = np.dot(T, G)
        sv = np.linalg.solve(C, q)
        return sv

    def optimise_fit6(self, f, S, N, Fseed, sv, loop_plan, Tol, quiet):
        """
        Iterative non-linear fit, NLQFIT6 Step (2).

        Optimised fit of Q-factor (QL) and resonant frequency (FL)
        by the gradient-descent method.

        Uses the results of the initial fit (sv) as the starting
        values for the iteration.

        Parameters
        ----------
        f : np.ndarray
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
            list of fitted parameters: [m1, m2, m3, m4, m5, m5 * Flwst / m6]
        weighting_ratio : float
        number_iterations : int
        RMS_Error : float

        References
        ----------
        .. [#] MAT 58, section 2.2, eqn. (22).

        """
        if loop_plan[-1] == "w":
            assert 0, "Last item in loop_plan must not be w (weight calculation)"
        if loop_plan[0] == "w":
            assert 0, "First item in loop_plan must not be w (weight calculation)"
        if loop_plan[-1] != "c":
            print("Warning: Last item in loop_plan is not c so convergence not tested!")
        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = sv[1] / sv[4]  # a''/QL
        m2 = -sv[0] / sv[4]
        m3 = sv[2] - m1
        m4 = sv[3] - m2
        m5 = sv[4]
        Flwst = f[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * m5 / Fseed
        last_op = "n"
        del sv
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in loop_plan:

            if op == "w":  # Fr                       QL
                PV = self.angular_weights(f, Flwst * float(m5) / float(m6), float(m5))
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if not quiet:
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
                M = np.zeros([N2, 6])  # X is the transpose of M
                G = np.zeros(N2)[:, np.newaxis]
                c1 = complex(-m4, m3)
                c2 = complex(m1, m2)
                c3 = complex(m3, m4)
                for i in range(N):
                    i2 = i + N
                    y = 1.0 / complex(1.0, 2 * (m6 * f[i] / Flwst - m5))
                    u = c1 * y * y * 2
                    u2 = -u * f[i] / Flwst
                    M[i, :] = 1.0, 0.0, y.real, -y.imag, u.real, u2.real
                    M[i2, :] = 0.0, 1.0, y.imag, y.real, u.imag, u2.imag
                    v = c2 + c3 * y
                    r = S[i] - v  # residual
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
                    den = complex(1.0, 2 * (m6 * f[i] / Flwst - m5))
                    ip = PV[i]
                    E = S[i] - complex(m1, m2) - complex(m3, m4) / den
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if not quiet:
                    if last_op == "c":
                        print(
                            "      Iteration %i, RMS_Error %10.8f"
                            % (iterations, RMS_Error)
                        )
                    else:
                        print(
                            "Op %c, Iteration %i, RMS_Error %10.8f"
                            % (op, iterations, RMS_Error)
                        )
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < Tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if not quiet:
                print()
        return (
            [m1, m2, m3, m4, m5, m5 * Flwst / m6],
            weighting_ratio,
            number_iterations,
            RMS_Error,
        )

    def optimise_fit7(self, F, S, N, Fseed, sv, loop_plan, Tol, quiet):
        """
        Iterative non-linear fit, NLQFIT7 Step (2).

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

        if loop_plan[-1] == "w":
            assert 0, "Last item in loop_plan must not be w (weight calculation)"
        if loop_plan[0] == "w":
            assert 0, "First item in loop_plan must not be w (weight calculation)"
        if loop_plan[-1] != "c":
            print("Warning: Last item in loop_plan is not c so convergence not tested!")

        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = sv[1] / sv[4]  # a''/QL
        m2 = -sv[0] / sv[4]
        m3 = sv[2] - m1
        m4 = sv[3] - m2
        m5 = sv[4]
        Flwst = F[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * m5 / Fseed
        m7 = 0.0
        last_op = "n"
        del sv
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in loop_plan:

            if op == "w":  # Fr                       QL
                PV = self.angular_weights(F, Flwst * float(m5) / float(m6), float(m5))
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if not quiet:
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
                    y = 1.0 / complex(1.0, 2 * (m6 * F[i] / Flwst - m5))
                    fdn = F[i] / Flwst - m5 / m6
                    pj = complex(0.0, m7 * fdn)
                    expm7 = np.exp(pj)
                    ym = y * expm7
                    u = c1 * y * ym * 2
                    u2 = -u * F[i] / Flwst
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
                    r = S[i] - v  # residual
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
                    fdn = F[i] / Flwst - m5 / m6
                    den = complex(1.0, 2 * (m6 * F[i] / Flwst - m5))
                    pj = complex(0.0, m7 * fdn)
                    E = S[i] - (c2 + c3 / den) * np.exp(pj)
                    ip = PV[i]
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if not quiet:
                    if last_op == "c":
                        print(
                            "      Iteration %i, RMS_Error %10.8f"
                            % (iterations, RMS_Error)
                        )
                    else:
                        print(
                            "Op %c, Iteration %i, RMS_Error %10.8f"
                            % (op, iterations, RMS_Error)
                        )
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < Tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if not quiet:
                print()
        return (
            [m1, m2, m3, m4, m5, m5 * Flwst / m6, m7 / Flwst],
            weighting_ratio,
            number_iterations,
            RMS_Error,
        )

    def optimise_fit8(self, F, S, N, Fseed, sv, loop_plan, Tol, quiet):
        """
        Iterative non-linear fit, NLQFIT8 Step (2).

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
        if loop_plan[-1] == "w":
            assert 0, "Last item in loop_plan must not be w (weight calculation)"
        if loop_plan[0] == "w":
            assert 0, "First item in loop_plan must not be w (weight calculation)"
        if loop_plan[-1] != "c":
            print("Warning: Last item in loop_plan is not c so convergence not tested!")

        N2 = N * 2
        iterations = 0
        PV = np.ones(N)  # default weights vector
        PV2 = np.ones(N2)

        m1 = sv[1] / sv[4]  # a''/QL
        m2 = -sv[0] / sv[4]
        m3 = sv[2] - m1
        m4 = sv[3] - m2
        m5 = sv[4]
        Flwst = F[0]  # lowest freq. is a convenient normalisation factor.
        m6 = Flwst * m5 / Fseed
        m8 = 0.0
        m9 = 0.0
        last_op = "n"
        del sv
        weighting_ratio = None
        number_iterations = 0

        ## Loop through all of the operations specified in loop_plan
        for op in loop_plan:

            if op == "w":  # Fr                       QL
                PV = self.angular_weights(F, Flwst * float(m5) / float(m6), float(m5))
                weighting_ratio = max(PV) / min(PV)
                PV2 = np.concatenate((PV, PV))
                if not quiet:
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
                    y = 1.0 / complex(1.0, 2 * (m6 * F[i] / Flwst - m5))
                    u = c1 * y * y * 2
                    u2 = -u * F[i] / Flwst
                    FL = Flwst * m5 / m6
                    t = 2 * (F[i] - FL) / FL
                    M[i, :] = 1.0, 0.0, y.real, -y.imag, u.real, u2.real, t, 0.0
                    M[i2, :] = 0.0, 1.0, y.imag, y.real, u.imag, u2.imag, 0.0, t
                    v = c2 + c3 * y + (m8 + 1j * m9) * t
                    r = S[i] - v  # residual
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
                    den = complex(1.0, 2 * (m6 * F[i] / Flwst - m5))
                    FL = Flwst * m5 / m6
                    t = 2 * (F[i] - FL) / FL
                    ip = PV[i]
                    E = (
                        S[i]
                        - complex(m1, m2)
                        - complex(m8, m9) * t
                        - complex(m3, m4) / den
                    )
                    SumNum = SumNum + ip * (E.real * E.real + E.imag * E.imag)
                    SumDen = SumDen + ip
                RMS_Error = np.sqrt(SumNum / SumDen)
                if not quiet:
                    if last_op == "c":
                        print(
                            "      Iteration %i, RMS_Error %10.8f"
                            % (iterations, RMS_Error)
                        )
                    else:
                        print(
                            "Op %c, Iteration %i, RMS_Error %10.8f"
                            % (op, iterations, RMS_Error)
                        )
                last_op = op

                if seek_convergence:
                    if Last_RMS_Error is not None:
                        delta_S = abs(RMS_Error - Last_RMS_Error)
                        TerminationConditionMet = delta_S < Tol
                else:
                    TerminationConditionMet = True
            # After last operation, we end up here ...
            if not quiet:
                print()
        return (
            [m1, m2, m3, m4, m8, m9, m5, m5 * Flwst / m6],
            weighting_ratio,
            number_iterations,
            RMS_Error,
        )

    def Q_circle(self, A, m1, m2, m3, m4):
        """
        Q-circle diameter.

        .. math::

            d = A|b + j a/Q_L|

        Parameters
        ----------
        A : float
            Correction factor magnitude
        m1 : float
            real part of cal_gamma_V
        m2 : float
            imag part of cal_gamma_V
        m3 : float
            real part of b + j a/Q_L
        m4 : float
            imag part of b + j a/Q_L

        Returns
        -------
        cal_diam : float
            calibrated Q-circle diameter d.
        cal_gamma_V : complex
            calibrated gamma_V
        cal_gamma_T : complex
            calibrated gamma_T = A.b

        References
        ----------
        .. [#] MAT 58, section 2.5, eqn. (31).

        """
        aqratio = complex(m1, m2)
        b = complex(m1 + m3, m2 + m4)
        cal_diam = abs(b - aqratio) * A
        cal_gamma_V = complex(m1, m2) * A
        cal_gamma_T = b * A
        return cal_diam, cal_gamma_V, cal_gamma_T

    def Q_unloaded(self, mv, scaling_factor_A, trmode, quiet):
        """
        Unloaded Q-factor and various 'calibrated' quantities.

        Parameters
        ----------
        mv : list
            solution produced by OptimiseFit.
        scaling_factor_A : float or str
            scaling factor as defined in MAT 58.
            For reflection_method1, can specify as 'AUTO'
            to use the magnitude of the fitted detuned reflection coefficient (gammaV)
        trmode : str
            'transmission', 'reflection_method1',
            'reflection_method2' or 'absorption'.
        quiet : bool
            Boolean flag controlling output of information to the console.

        Raises
        ------
        ValueError
            If the mv parameter has an incorrect length.

        Returns
        -------
        results : list
            (Q0, cal_diam, cal_gamma_V, cal_gamma_T)

        References
        ----------
        .. "Q-factor Measurement by using a Vector Network Analyser",
            A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)
            https://eprintspublications.npl.co.uk/9304/
        .. "The Physics of Superconducting Microwave Resonators"
            Gao, Jiansong (2008), doi:10.7907/RAT0-VM75.
            https://resolver.caltech.edu/CaltechETD:etd-06092008-235549

        """
        if type(scaling_factor_A) is str:
            if scaling_factor_A.upper() == "AUTO":
                auto_flag = True
            else:
                raise ValueError("Illegal Scaling factor; should be a float or 'AUTO'")
        else:
            # Also permit negative scaling factor to indicate 'Auto'
            if scaling_factor_A < 0.0:
                auto_flag = True
            else:
                auto_flag = False

        if len(mv) == 6:
            m1, m2, m3, m4, m5, FL = mv
        elif len(mv) == 7:
            m1, m2, m3, m4, m5, FL, m7_flwst = mv
        elif len(mv) == 8:
            m1, m2, m3, m4, m8, m9, m5, FL = mv
        else:
            raise ValueError('Unkown list length for mv')

        if trmode == "transmission":
            if auto_flag:
                raise ValueError('Scaling factor must not be "Auto" for transmission case')
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(
                scaling_factor_A, m1, m2, m3, m4
            )
            if cal_diam == 1.0:
                raise ZeroDivisionError("Divide by zero forestalled in calculation of Qo")
            Q0 = m5 / (1.0 - cal_diam)

        elif trmode == "reflection_method1":
            if auto_flag:
                if not quiet:
                    print(
                        'Supplied scaling_factor_A is "Auto", so using fitted data to estimate it'
                    )
                scaling_factor_A = 1.0 / abs(
                    complex(m1, m2)
                )  # scale to gammaV if 'AUTO'
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(
                scaling_factor_A, m1, m2, m3, m4
            )
            cal_touching_circle_diam = 2.0
            if not quiet:
                print(
                    "  Q-circle diam = %5.3f, touching_circle_diam = %5.3f"
                    % (cal_diam, cal_touching_circle_diam)
                )
            den = cal_touching_circle_diam / cal_diam - 1.0
            Q0 = m5 * (1.0 + 1.0 / den)

        elif trmode == "reflection_method2":
            if auto_flag:
                raise ValueError('Scaling factor must not be "Auto" for Method 2')
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(
                scaling_factor_A, m1, m2, m3, m4
            )
            gv = abs(cal_gamma_V)
            gv2 = gv * gv
            mb = abs(cal_gamma_T)
            cosphi = (gv2 + cal_diam * cal_diam - mb * mb) / (
                2.0 * gv * cal_diam
            )  # Cosine rule
            cal_touching_circle_diam = (1.0 - gv2) / (1.0 - gv * cosphi)
            if not quiet:
                print(
                    "  Q-circle diam = %5.3f, touching_circle_diam = %5.3f"
                    % (cal_diam, cal_touching_circle_diam)
                )
            den = cal_touching_circle_diam / cal_diam - 1.0
            Q0 = m5 * (1.0 + 1.0 / den)

        elif trmode == "notch" or trmode == "absorption":  # By transmission
            if auto_flag:
                if not quiet:
                    print(
                        'Notch/absorption Qo calculation: Supplied scaling_factor_A is "Auto", so using fitted data to estimate it'
                    )
                scaling_factor_A = 1.0 / abs(
                    complex(m1, m2)
                )  # scale to gammaV if 'AUTO'
            cal_diam, cal_gamma_V, cal_gamma_T = self.Q_circle(
                scaling_factor_A, m1, m2, m3, m4
            )
            if not quiet:
                print("  Q-circle diam = %5.3f" % (cal_diam))
            if cal_diam == 1.0:
                raise ZeroDivisionError("Divide by zero forestalled in calculation of Qo")
            den = 1.0 / cal_diam - 1.0  # Gao thesis (2008) 4.35 and 4.40
            Q0 = m5 * (
                1.0 + 1.0 / den
            )  # https://resolver.caltech.edu/CaltechETD:etd-06092008-235549
            # For this type of resonator, critical coupling occurs for cal_diam = 0.5.
        else:
            raise ValueError("Unknown trmode")

        return Q0, cal_diam, cal_gamma_V, cal_gamma_T
