from __future__ import annotations

import logging
import os
import warnings
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import trapezoid

from .util import Axes, axes_kwarg

# imports for type hinting
if TYPE_CHECKING:
    from .network import Network


logger = logging.getLogger(__name__)


class VectorFitting:
    """
    This class provides a Python implementation of the Vector Fitting algorithm and various functions for the fit
    analysis, passivity evaluation and enforcement, and export of SPICE equivalent circuits.

    Parameters
    ----------
    network : :class:`skrf.network.Network`
            Network instance of the :math:`N`-port holding the frequency responses to be fitted, for example a
            scattering, impedance or admittance matrix.

    Examples
    --------
    Load the `Network`, create a `VectorFitting` instance, perform the fit with a given number of real and
    complex-conjugate starting poles:

    >>> nw_3port = skrf.Network('my3port.s3p')
    >>> vf = skrf.VectorFitting(nw_3port)
    >>> vf.vector_fit(n_poles_real=1, n_poles_cmplx=4)

    Notes
    -----
    The fitting code is based on the original algorithm [#Gustavsen_vectfit]_ and on two improvements for relaxed pole
    relocation [#Gustavsen_relaxed]_ and efficient (fast) solving [#Deschrijver_fast]_. See also the Vector Fitting
    website [#vectfit_website]_ for further information and download of the papers listed below. A Matlab implementation
    is also available there for reference.

    References
    ----------
    .. [#Gustavsen_vectfit] B. Gustavsen, A. Semlyen, "Rational Approximation of Frequency Domain Responses by Vector
        Fitting", IEEE Transactions on Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999,
        DOI: https://doi.org/10.1109/61.772353

    .. [#Gustavsen_relaxed] B. Gustavsen, "Improving the Pole Relocating Properties of Vector Fitting", IEEE
        Transactions on Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006,
        DOI: https://doi.org/10.1109/TPWRD.2005.860281

    .. [#Deschrijver_fast] D. Deschrijver, M. Mrozowski, T. Dhaene, D. De Zutter, "Marcomodeling of Multiport Systems
        Using a Fast Implementation of the Vector Fitting Method", IEEE Microwave and Wireless Components Letters,
        vol. 18, no. 6, pp. 383-385, June 2008, DOI: https://doi.org/10.1109/LMWC.2008.922585

    .. [#vectfit_website] Vector Fitting website: https://www.sintef.no/projectweb/vectorfitting/
    """

    def __init__(self, network: Network):
        self.network = network
        """ Instance variable holding the Network to be fitted. This is the Network passed during initialization,
        which may be changed or set to *None*. """

        self.poles = None
        """ Instance variable holding the list of fitted poles. Will be initialized by :func:`vector_fit`. """

        self.residues = None
        """ Instance variable holding the list of fitted residues. Will be initialized by :func:`vector_fit`. """

        self.proportional_coeff = None
        """ Instance variable holding the list of fitted proportional coefficients. Will be initialized by
        :func:`vector_fit`. """

        self.constant_coeff = None
        """ Instance variable holding the list of fitted constants. Will be initialized by :func:`vector_fit`. """

        self.max_iterations = 100
        """ Instance variable specifying the maximum number of iterations for the fitting process and for the passivity
        enforcement. To be changed by the user before calling :func:`vector_fit` and/or :func:`passivity_enforce`. """

        self.max_tol = 1e-6
        """ Instance variable specifying the convergence criterion in terms of relative tolerance. To be changed by the
         user before calling :func:`vector_fit`. """

        self.wall_clock_time = 0
        """ Instance variable holding the wall-clock time (in seconds) consumed by the most recent fitting process with
        :func:`vector_fit`. Subsequent calls of :func:`vector_fit` will overwrite this value. """

        self.d_res_history = []
        self.delta_max_history = []
        self.history_max_sigma = []
        self.history_cond_A = []
        self.history_rank_deficiency = []

    @staticmethod
    def get_spurious(poles: np.ndarray, residues: np.ndarray, n_freqs: int = 101, gamma: float = 0.03) -> np.ndarray:
        """
        Classifies fitted pole-residue pairs as spurious or not spurious. The implementation is based on the evaluation
        of band-limited energy norms (p=2) of the resonance curves of individual pole-residue pairs, as proposed in
        [#Grivet-Talocia]_.

        Parameters
        ----------
        poles : ndarray, shape (N)
            Array of fitted poles

        residues : ndarray, shape (M, N)
            Array of fitted residues

        n_freqs : int, optional
            Number of frequencies for the evaluation. The frequency range is chosen automatically and the default
            101 frequencies should be appropriate in most cases.

        gamma : float, optional
            Sensitivity threshold for the classification. Typical values range from 0.01 to 0.05.

        Returns
        -------
        ndarray, bool, shape (M)
            Boolean array having the same shape as :attr:`poles`. `True` marks the respective pole as spurious.

        References
        ----------
        .. [#Grivet-Talocia] S. Grivet-Talocia and M. Bandinu, "Improving the convergence of vector fitting for
            equivalent circuit extraction from noisy frequency responses," in IEEE Transactions on Electromagnetic
            Compatibility, vol. 48, no. 1, pp. 104-120, Feb. 2006, DOI: https://doi.org/10.1109/TEMC.2006.870814
        """

        omega_eval = np.linspace(np.min(poles.imag) / 3, np.max(poles.imag) * 3, n_freqs)
        h = (residues[:, None, :] / (1j * omega_eval[:, None] - poles)
             + np.conj(residues[:, None, :]) / (1j * omega_eval[:, None] - np.conj(poles)))
        norm2 = np.sqrt(trapezoid(h.real ** 2 + h.imag ** 2, omega_eval, axis=1))
        spurious = np.all(norm2 / np.mean(norm2) < gamma, axis=0)
        return spurious

    @staticmethod
    def get_model_order(poles: np.ndarray) -> int:
        """
        Returns the model order calculated with :math:`N_{real} + 2 N_{complex}` for a given set of poles.

        Parameters
        ----------
        poles: ndarray
            The poles of the model as a list or NumPy array.

        Returns
        -------
        order: int
        """
        # poles.imag != 0 is True(1) for complex poles, False (0) for real poles.
        # Adding one to each element gives 2 columns for complex and 1 column for real poles.
        return np.sum((poles.imag != 0) + 1)

    def vector_fit(self, n_poles_real: int = 2, n_poles_cmplx: int = 2, init_pole_spacing: str = 'lin',
                   parameter_type: str = 's', fit_constant: bool = True, fit_proportional: bool = False) -> None:
        """
        Main work routine performing the vector fit. The results will be stored in the class variables
        :attr:`poles`, :attr:`residues`, :attr:`proportional_coeff` and :attr:`constant_coeff`.

        Parameters
        ----------
        n_poles_real : int, optional
            Number of initial real poles. See notes.

        n_poles_cmplx : int, optional
            Number of initial complex conjugate poles. See notes.

        init_pole_spacing : str, optional
            Type of initial pole spacing across the frequency interval of the S-matrix. Either *linear* (`'lin'`),
            *logarithmic* (`'log'`), or `custom`. In case of `custom`, the initial poles must be stored in :attr:`poles`
            as a NumPy array before calling this method. They will be overwritten by the final poles. The
            initialization parameters `n_poles_real` and `n_poles_cmplx` will be ignored in case of `'custom'`.

        parameter_type : str, optional
            Representation type of the frequency responses to be fitted. Either *scattering* (`'s'` or `'S'`),
            *impedance* (`'z'` or `'Z'`) or *admittance* (`'y'` or `'Y'`). It's recommended to perform the fit on the
            original S parameters. Otherwise, scikit-rf will convert the responses from S to Z or Y, which might work
            for the fit but can cause other issues.

        fit_constant : bool, optional
            Include a constant term **d** in the fit.

        fit_proportional : bool, optional
            Include a proportional term **e** in the fit.

        Returns
        -------
        None
            No return value.

        Notes
        -----
        The required number of real or complex conjugate starting poles depends on the behaviour of the frequency
        responses. To fit a smooth response such as a low-pass characteristic, 1-3 real poles and no complex conjugate
        poles is usually sufficient. If resonances or other types of peaks are present in some or all of the responses,
        a similar number of complex conjugate poles is required. Be careful not to use too many poles, as excessive
        poles will not only increase the computation workload during the fitting and the subsequent use of the model,
        but they can also introduce unwanted resonances at frequencies well outside the fit interval.

        See Also
        --------
        auto_fit : Automatic vector fitting routine with pole adding and skimming.
        """

        timer_start = timer()

        # use normalized frequencies during the iterations (seems to be more stable during least-squares fit)
        norm = np.average(self.network.f)
        # norm = np.exp(np.mean(np.log(self.network.f)))
        freqs_norm = np.array(self.network.f) / norm

        # get initial poles
        poles = self._init_poles(freqs_norm, n_poles_real, n_poles_cmplx, init_pole_spacing)

        # check and normalize custom poles
        if poles is None:
            if self.poles is not None and len(self.poles) > 0:
                poles = self.poles / norm
            else:
                raise ValueError('Initial poles must be provided in `self.poles` when calling with '
                                 '`init_pole_spacing == \'custom\'`.')

        # save initial poles (un-normalize first)
        initial_poles = poles * norm
        max_singular = 1

        logger.info('### Starting pole relocation process.\n')

        # select network representation type
        if parameter_type.lower() == 's':
            nw_responses = self.network.s
        elif parameter_type.lower() == 'z':
            nw_responses = self.network.z
        elif parameter_type.lower() == 'y':
            nw_responses = self.network.y
        else:
            warnings.warn('Invalid choice of matrix parameter type (S, Z, or Y); proceeding with scattering '
                          'representation.', UserWarning, stacklevel=2)
            nw_responses = self.network.s

        # stack frequency responses as a single vector
        # stacking order (row-major):
        # s11, s12, s13, ..., s21, s22, s23, ...
        freq_responses = []
        for i in range(self.network.nports):
            for j in range(self.network.nports):
                freq_responses.append(nw_responses[:, i, j])
        freq_responses = np.array(freq_responses)

        # responses will be weighted according to their norm;
        # alternative: equal weights with weight_response = 1.0
        # or anti-proportional weights with weight_response = 1 / np.linalg.norm(freq_response)
        weights_responses = np.linalg.norm(freq_responses, axis=1)
        #weights_responses = np.ones(self.network.nports ** 2)
        #weights_responses = 10 / np.exp(np.mean(np.log(np.abs(freq_responses)), axis=1))

        # ITERATIVE FITTING OF POLES to the provided frequency responses
        # initial set of poles will be replaced with new poles after every iteration
        iterations = self.max_iterations
        self.d_res_history = []
        self.delta_max_history = []
        self.history_cond_A = []
        self.history_rank_deficiency = []
        converged = False

        # POLE RELOCATION LOOP
        while iterations > 0:
            logger.info(f'Iteration {self.max_iterations - iterations + 1}')

            poles, d_res, cond, rank_deficiency, residuals, singular_vals = self._pole_relocation(
                poles, freqs_norm, freq_responses, weights_responses, fit_constant, fit_proportional)

            logger.info(f'Condition number of coefficient matrix is {int(cond)}')
            self.history_cond_A.append(cond)

            self.history_rank_deficiency.append(rank_deficiency)
            logger.info(f'Rank deficiency is {rank_deficiency}.')

            self.d_res_history.append(d_res)
            logger.info(f'd_res = {d_res}')

            # calculate relative changes in the singular values; stop iteration loop once poles have converged
            new_max_singular = np.amax(singular_vals)
            delta_max = np.abs(1 - new_max_singular / max_singular)
            self.delta_max_history.append(delta_max)
            logger.info(f'Max. relative change in residues = {delta_max}\n')
            max_singular = new_max_singular

            stop = False
            if delta_max < self.max_tol:
                if converged:
                    # is really converged, finish
                    logger.info(f'Pole relocation process converged after {self.max_iterations - iterations + 1} '
                                  'iterations.')
                    stop = True
                else:
                    # might be converged, but do one last run to be sure
                    converged = True
            else:
                if converged:
                    # is not really converged, continue
                    converged = False

            iterations -= 1

            if iterations == 0:
                # loop ran into iterations limit; trying to assess the issue
                max_cond = np.amax(self.history_cond_A)
                max_deficiency = np.amax(self.history_rank_deficiency)
                if max_cond > 1e10:
                    hint_illcond = ('\nHint: the linear system was ill-conditioned (max. condition number was '
                                    f'{max_cond}).')
                else:
                    hint_illcond = ''
                if max_deficiency < 0:
                    hint_rank = ('\nHint: the coefficient matrix was rank-deficient (max. rank deficiency was '
                                 f'{max_deficiency}).')
                else:
                    hint_rank = ''
                if converged and stop is False:
                    warnings.warn('Vector Fitting: The pole relocation process barely converged to tolerance. '
                                  f'It took the max. number of iterations (N_max = {self.max_iterations}). '
                                  'The results might not have converged properly.'
                                  + hint_illcond + hint_rank, RuntimeWarning, stacklevel=2)
                else:
                    warnings.warn('Vector Fitting: The pole relocation process stopped after reaching the '
                                  f'maximum number of iterations (N_max = {self.max_iterations}). '
                                  'The results did not converge properly.'
                                  + hint_illcond + hint_rank, RuntimeWarning, stacklevel=2)

            if stop:
                iterations = 0

        # ITERATIONS DONE
        logger.info('Initial poles before relocation:')
        logger.info(initial_poles)

        logger.info('Final poles:')
        logger.info(poles * norm)

        logger.info('\n### Starting residues calculation process.\n')

        # finally, solve for the residues with the previously calculated poles
        residues, constant_coeff, proportional_coeff, residuals, rank, singular_vals = self._fit_residues(
            poles, freqs_norm, freq_responses, fit_constant, fit_proportional)

        # save poles, residues, d, e in actual frequencies (un-normalized)
        self.poles = poles * norm
        self.residues = np.array(residues) * norm
        self.constant_coeff = np.array(constant_coeff)
        self.proportional_coeff = np.array(proportional_coeff) / norm

        timer_stop = timer()
        self.wall_clock_time = timer_stop - timer_start

        logger.info(f'\n### Vector fitting finished in {self.wall_clock_time} seconds.\n')

        # raise a warning if the fitted Network is passive but the fit is not (only without proportional_coeff):
        if self.network.is_passive() and not fit_proportional:
            if not self.is_passive():
                warnings.warn('The fitted network is passive, but the vector fit is not passive. Consider running '
                              '`passivity_enforce()` to enforce passivity before using this model.',
                              UserWarning, stacklevel=2)

    def auto_fit(self, n_poles_init_real: int = 3, n_poles_init_cmplx: int = 3, n_poles_add: int = 3,
                 model_order_max: int = 100, iters_start: int = 3, iters_inter: int = 3, iters_final: int = 5,
                 target_error: float = 1e-2, alpha: float = 0.03, gamma: float = 0.03, nu_samples: float = 1.0,
                 parameter_type: str = 's') -> (np.ndarray, np.ndarray):
        """
        Automatic fitting routine implementing the "vector fitting with adding and skimming" algorithm as proposed in
        [#Grivet-Talocia]_. This algorithm is able to provide high quality macromodels with automatic model order
        optimization, while improving both the rate of convergence and the fit quality in case of noisy data.
        The resulting model parameters will be stored in the class variables :attr:`poles`, :attr:`residues`,
        :attr:`proportional_coeff` and :attr:`constant_coeff`.

        Parameters
        ----------
        n_poles_init_real: int, optional
            Number of real poles in the initial model.

        n_poles_init_cmplx: int, optional
            Number of complex conjugate poles in the initial model.

        n_poles_add: int, optional
            Number of new poles allowed to be added in each refinement iteration, if possible. This controls how fast
            the model order is allowed to grow. Unnecessary poles will have to be skimmed and removed later. This
            parameter has a strong effect on the convergence.

        model_order_max: int, optional
            Maximum model order as calculated with :math:`N_{real} + 2 N_{complex}`. This parameter provides a stopping
            criterion in case the refinement process is not converging.

        iters_start: int, optional
            Number of initial iterations for pole relocations as in regular vector fitting.

        iters_inter: int, optional
            Number of intermediate iterations for pole relocations during each iteration of the refinement process.

        iters_final: int, optional
            Number of final iterations for pole relocations after the refinement process.

        target_error: float, optional
            Target for the model error to be reached during the refinement process. The actual achievable error is
            bound by the noise in the data. If specified with a number greater than the noise floor, this parameter
            provides another stopping criterion for the refinement process. It therefore affects both the convergence,
            the final error, and the final model order (number of poles used in the model).

        alpha: float, optional
            Threshold for the error decay to stop the refinement loop in case of error stagnation. This parameter
            provides another stopping criterion for cases where the model already has enough poles but the target error
            still cannot be reached because of excess noise (target error too small for noise level in the data).

        gamma: float, optional
            Threshold for the detection of spurious poles.

        nu_samples: float, optional
            Required and enforced (relative) spacing in terms of frequency samples between existing poles and
            relocated or added poles. The number can be a float, it does not have to be an integer.

        parameter_type: str, optional
            Representation type of the frequency responses to be fitted. Either *scattering* (`'s'` or `'S'`),
            *impedance* (`'z'` or `'Z'`) or *admittance* (`'y'` or `'Y'`). It's recommended to perform the fit on the
            original S parameters. Otherwise, scikit-rf will convert the responses from S to Z or Y, which might work
            for the fit but can cause other issues.

        Returns
        -------
        None
            No return value.

        See Also
        --------
        vector_fit : Regular vector fitting routine.

        References
        ----------
        .. [#Grivet-Talocia] S. Grivet-Talocia and M. Bandinu, "Improving the convergence of vector fitting for
            equivalent circuit extraction from noisy frequency responses," in IEEE Transactions on Electromagnetic
            Compatibility, vol. 48, no. 1, pp. 104-120, Feb. 2006, DOI: https://doi.org/10.1109/TEMC.2006.870814
        """

        self.d_res_history = []
        self.delta_max_history = []
        self.history_cond_A = []
        self.history_rank_deficiency = []
        max_singular = 1
        error_peak_history = []
        model_order_history = []

        timer_start = timer()

        # use normalized frequencies during the iterations (seems to be more stable during least-squares fit)
        norm = np.average(self.network.f)
        # norm = np.exp(np.mean(np.log(self.network.f)))
        freqs_norm = np.array(self.network.f) / norm
        omega_norm = 2 * np.pi * freqs_norm
        nu = (omega_norm[1] - omega_norm[0]) * nu_samples

        # get initial poles
        poles = self._init_poles(freqs_norm, n_poles_init_real, n_poles_init_cmplx, 'lin')

        logger.info('### Starting pole relocation process.\n')

        # select network representation type
        if parameter_type.lower() == 's':
            nw_responses = self.network.s
            fit_constant = True
            fit_proportional = False
        elif parameter_type.lower() == 'z':
            nw_responses = self.network.z
            fit_constant = True
            fit_proportional = True
        elif parameter_type.lower() == 'y':
            nw_responses = self.network.y
            fit_constant = True
            fit_proportional = True
        else:
            warnings.warn('Invalid choice of matrix parameter type (S, Z, or Y); proceeding with scattering '
                          'representation.', UserWarning, stacklevel=2)
            nw_responses = self.network.s
            fit_constant = True
            fit_proportional = False

        # stack frequency responses as a single vector
        # stacking order (row-major):
        # s11, s12, s13, ..., s21, s22, s23, ...
        freq_responses = []
        for i in range(self.network.nports):
            for j in range(self.network.nports):
                freq_responses.append(nw_responses[:, i, j])
        freq_responses = np.array(freq_responses)

        # responses will be weighted according to their norm;
        # alternative: equal weights with weight_response = 1.0
        # or anti-proportional weights with weight_response = 1 / np.linalg.norm(freq_response)
        weights_responses = np.linalg.norm(freq_responses, axis=1)
        # weights_responses = np.ones(self.network.nports ** 2)
        # weights_responses = 10 / np.exp(np.mean(np.log(np.abs(freq_responses)), axis=1))

        # INITIAL POLE RELOCATION FOR i_start ITERATIONS
        for _ in range(iters_start):
            poles, d_res, cond, rank_deficiency, residuals, singular_vals = self._pole_relocation(
                poles, freqs_norm, freq_responses, weights_responses, fit_constant, fit_proportional)

            self.d_res_history.append(d_res)

            logger.info(f'Condition number of coefficient matrix is {int(cond)}')
            self.history_cond_A.append(cond)

            self.history_rank_deficiency.append(rank_deficiency)
            logger.info(f'Rank deficiency is {rank_deficiency}.')

            new_max_singular = np.amax(singular_vals)
            delta_max = np.abs(1 - new_max_singular / max_singular)
            self.delta_max_history.append(delta_max)
            logger.info(f'Max. relative change in residues = {delta_max}\n')
            max_singular = new_max_singular

        # RESIDUE FITTING FOR ERROR COMPUTATION
        residues, constant_coeff, proportional_coeff, residuals, rank, singular_vals = self._fit_residues(
            poles, freqs_norm, freq_responses, fit_constant, fit_proportional, enforce_dc=False)
        delta = self._get_delta(poles, residues, constant_coeff, proportional_coeff, freqs_norm, freq_responses,
                                weights_responses)
        error_peak = np.max(delta)
        error_peak_history.append(error_peak)

        model_order = self.get_model_order(poles)
        model_order_history.append(model_order)

        delta_eps = 10 * alpha

        # POLE SKIMMING AND ADDING LOOP
        while error_peak > target_error and model_order < model_order_max and delta_eps > alpha:

            # SKIMMING OF SPURIOUS POLES
            spurious = self.get_spurious(poles, residues, gamma=gamma)
            n_skim = np.sum(spurious)
            poles = poles[~spurious]

            # REPLACING SPURIOUS POLE AND ADDING NEW POLES
            idx_freqs_start, idx_freqs_stop, idx_freqs_max, delta_mean_bands = self._find_error_bands(freqs_norm, delta)

            n_bands = len(idx_freqs_max)
            if n_bands < n_skim:
                n_add = n_bands
            elif n_bands < n_skim + n_poles_add:
                n_add = n_bands
            else:
                n_add = n_skim + n_poles_add

            for i in range(n_add):
                omega_add = omega_norm[idx_freqs_max[i]]
                pole_add = (-0.01 + 1j) * omega_add

                # compute distance to neighbouring poles
                abs_poles_existing = np.abs(poles) - pole_add.imag  # (equation 16)
                #abs_poles_existing = np.abs(poles - pole_add)   # (equation 17)

                # avoid forbidden bands (too close to neighbour)
                if np.min(abs_poles_existing) < nu or pole_add.imag < nu:
                    # decide shift direction (towards higher or lower frequencies)
                    if idx_freqs_max[i] > 0:
                        delta_below = delta[idx_freqs_max[i] - 1]
                    else:
                        delta_below = 0
                    if idx_freqs_max[i] < len(omega_norm) - 1:
                        delta_above = delta[idx_freqs_max[i] + 1]
                    else:
                        delta_above = 0

                    if delta_above > delta_below:
                        # shift to higher frequencies
                        pole_add += 1j * nu
                    else:
                        # shift to lower frequencies
                        pole_add -= 1j * nu

                poles = np.append(poles, [pole_add])

            # INTERMEDIATE POLE RELOCATION FOR i_inter ITERATIONS
            for _ in range(iters_inter):
                poles, d_res, cond, rank_deficiency, residuals, singular_vals = self._pole_relocation(
                    poles, freqs_norm, freq_responses, weights_responses, fit_constant, fit_proportional)

                self.d_res_history.append(d_res)

                logger.info(f'Condition number of coefficient matrix is {int(cond)}')
                self.history_cond_A.append(cond)

                self.history_rank_deficiency.append(rank_deficiency)
                logger.info(f'Rank deficiency is {rank_deficiency}.')

                new_max_singular = np.amax(singular_vals)
                delta_max = np.abs(1 - new_max_singular / max_singular)
                self.delta_max_history.append(delta_max)
                logger.info(f'Max. relative change in residues = {delta_max}\n')
                max_singular = new_max_singular

            # RESIDUE FITTING FOR ERROR COMPUTATION
            residues, constant_coeff, proportional_coeff, residuals, rank, singular_vals = self._fit_residues(
                poles, freqs_norm, freq_responses, fit_constant, fit_proportional, enforce_dc=False)
            delta = self._get_delta(poles, residues, constant_coeff, proportional_coeff, freqs_norm, freq_responses,
                                    weights_responses)
            error_peak_history.append(np.max(delta))

            m = 3
            if len(error_peak_history) > m:
                delta_eps = np.mean(np.abs(np.diff(error_peak_history[-1-m:-1])))
            else:
                delta_eps = 1

            model_order = self.get_model_order(poles)
            model_order_history.append(model_order)

        # SKIMMING OF SPURIOUS POLES
        spurious = self.get_spurious(poles, residues, gamma=gamma)
        poles = poles[~spurious]

        # FINAL POLE RELOCATION FOR i_final ITERATIONS
        for _ in range(iters_final):
            poles, d_res, cond, rank_deficiency, residuals, singular_vals = self._pole_relocation(
                poles, freqs_norm, freq_responses, weights_responses, fit_constant, fit_proportional)

            self.d_res_history.append(d_res)

            logger.info(f'Condition number of coefficient matrix is {int(cond)}')
            self.history_cond_A.append(cond)

            self.history_rank_deficiency.append(rank_deficiency)
            logger.info(f'Rank deficiency is {rank_deficiency}.')

            new_max_singular = np.amax(singular_vals)
            delta_max = np.abs(1 - new_max_singular / max_singular)
            self.delta_max_history.append(delta_max)
            logger.info(f'Max. relative change in residues = {delta_max}\n')
            max_singular = new_max_singular

        # FINAL RESIDUE FITTING
        residues, constant_coeff, proportional_coeff, residuals, rank, singular_vals = self._fit_residues(
            poles, freqs_norm, freq_responses, fit_constant, fit_proportional, enforce_dc=True)

        # save poles, residues, d, e in actual frequencies (un-normalized)
        self.poles = poles * norm
        self.residues = np.array(residues) * norm
        self.constant_coeff = np.array(constant_coeff)
        self.proportional_coeff = np.array(proportional_coeff) / norm

        timer_stop = timer()
        self.wall_clock_time = timer_stop - timer_start

    @staticmethod
    def _init_poles(freqs: list, n_poles_real: int, n_poles_cmplx: int, init_pole_spacing: str):
        # create initial poles and space them across the frequencies in the provided Touchstone file

        fmin = np.amin(freqs)
        fmax = np.amax(freqs)

        # poles cannot be at f=0; hence, f_min for starting pole must be greater than 0
        if fmin == 0.0:
            # random choice: use 1/1000 of first non-zero frequency
            fmin = freqs[1] / 1000

        init_pole_spacing = init_pole_spacing.lower()
        if init_pole_spacing == 'log':
            pole_freqs_real = np.geomspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.geomspace(fmin, fmax, n_poles_cmplx)
        elif init_pole_spacing == 'lin':
            pole_freqs_real = np.linspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.linspace(fmin, fmax, n_poles_cmplx)
        elif init_pole_spacing == 'custom':
            pole_freqs_real = None
            pole_freqs_cmplx = None
        else:
            warnings.warn('Invalid choice of initial pole spacing; proceeding with linear spacing.',
                          UserWarning, stacklevel=2)
            pole_freqs_real = np.linspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.linspace(fmin, fmax, n_poles_cmplx)

        if pole_freqs_real is not None and pole_freqs_cmplx is not None:
            # init poles array of correct length
            poles = np.zeros(n_poles_real + n_poles_cmplx, dtype=complex)

            # add real poles
            for i, f in enumerate(pole_freqs_real):
                omega = 2 * np.pi * f
                poles[i] = -1 * omega

            # add complex-conjugate poles (store only positive imaginary parts)
            i_offset = len(pole_freqs_real)
            for i, f in enumerate(pole_freqs_cmplx):
                omega = 2 * np.pi * f
                poles[i_offset + i] = (-0.01 + 1j) * omega

            return poles

        else:
            return None

    @staticmethod
    def _pole_relocation(poles, freqs, freq_responses, weights_responses, fit_constant, fit_proportional):
        n_responses, n_freqs = np.shape(freq_responses)
        n_samples = n_responses * n_freqs
        omega = 2 * np.pi * freqs
        s = 1j * omega

        # weight of extra equation to avoid trivial solution
        weight_extra = np.linalg.norm(weights_responses[:, None] * freq_responses) / n_samples

        # weights w are applied directly to the samples, which get squared during least-squares fitting; hence sqrt(w)
        weights_responses = np.sqrt(weights_responses)
        weight_extra = np.sqrt(weight_extra)

        # count number of rows and columns in final coefficient matrix to solve for (c_res, d_res)
        # (ratio #real/#complex poles might change during iterations)

        # We need two columns for complex poles and one column for real poles in A matrix.
        # This number equals the model order.
        n_cols_unused = VectorFitting.get_model_order(poles)

        n_cols_used = n_cols_unused
        n_cols_used += 1
        idx_constant = []
        idx_proportional = []
        if fit_constant:
            idx_constant = [n_cols_unused]
            n_cols_unused += 1
        if fit_proportional:
            idx_proportional = [n_cols_unused]
            n_cols_unused += 1

        real_mask = poles.imag == 0
        # list of indices in 'poles' with real values
        idx_poles_real = np.nonzero(real_mask)[0]
        # list of indices in 'poles' with complex values
        idx_poles_complex = np.nonzero(~real_mask)[0]

        # positions (columns) of coefficients for real and complex-conjugate terms in the rows of A determine the
        # respective positions of the calculated residues in the results vector.
        # to have them ordered properly for the subsequent assembly of the test matrix H for eigenvalue extraction,
        # place real poles first, then complex-conjugate poles with their respective real and imaginary parts:
        # [r1', r2', ..., (r3', r3''), (r4', r4''), ...]
        n_real = len(idx_poles_real)
        n_cmplx = len(idx_poles_complex)
        idx_res_real = np.arange(n_real)
        idx_res_complex_re = n_real + 2 * np.arange(n_cmplx)
        idx_res_complex_im = idx_res_complex_re + 1

        # complex coefficient matrix of shape [N_responses, N_freqs, n_cols_unused + n_cols_used]
        # layout of each row:
        # [pole1, pole2, ..., (constant), (proportional), pole1, pole2, ..., constant]
        A = np.empty((n_responses, n_freqs, n_cols_unused + n_cols_used), dtype=complex)

        # calculate coefficients for real and complex residues in the solution vector
        #
        # real pole-residue term (r = r', p = p'):
        # fractional term is r' / (s - p')
        # coefficient for r' is 1 / (s - p')
        coeff_real = 1 / (s[:, None] - poles[None, idx_poles_real])

        # complex-conjugate pole-residue pair (r = r' + j r'', p = p' + j p''):
        # fractional term is r / (s - p) + conj(r) / (s - conj(p))
        #                   = [1 / (s - p) + 1 / (s - conj(p))] * r' + [1j / (s - p) - 1j / (s - conj(p))] * r''
        # coefficient for r' is 1 / (s - p) + 1 / (s - conj(p))
        # coefficient for r'' is 1j / (s - p) - 1j / (s - conj(p))
        coeff_complex_re = (1 / (s[:, None] - poles[None, idx_poles_complex]) +
                            1 / (s[:, None] - np.conj(poles[None, idx_poles_complex])))
        coeff_complex_im = (1j / (s[:, None] - poles[None, idx_poles_complex]) -
                            1j / (s[:, None] - np.conj(poles[None, idx_poles_complex])))

        # part 1: first sum of rational functions (variable c)
        A[:, :, idx_res_real] = coeff_real
        A[:, :, idx_res_complex_re] = coeff_complex_re
        A[:, :, idx_res_complex_im] = coeff_complex_im

        # part 2: constant (variable d) and proportional term (variable e)
        A[:, :, idx_constant] = 1
        A[:, :, idx_proportional] = s[:, None]

        # part 3: second sum of rational functions multiplied with frequency response (variable c_res)
        A[:, :, n_cols_unused + idx_res_real] = -1 * freq_responses[:, :, None] * coeff_real
        A[:, :, n_cols_unused + idx_res_complex_re] = -1 * freq_responses[:, :, None] * coeff_complex_re
        A[:, :, n_cols_unused + idx_res_complex_im] = -1 * freq_responses[:, :, None] * coeff_complex_im

        # part 4: constant (variable d_res)
        A[:, :, -1] = -1 * freq_responses

        A_ri = np.hstack((A.real, A.imag))

        # calculation of matrix sizes after QR decomposition:
        # stacked coefficient matrix (A.real, A.imag) has shape (L, M, N)
        # with
        # L = n_responses = n_ports ** 2
        # M = 2 * n_freqs (because of hstack with 2x n_freqs)
        # N = n_cols_unused + n_cols_used
        # then
        # R has shape (L, K, N) with K = min(M, N)
        dim_m = 2 * n_freqs
        dim_n = n_cols_unused + n_cols_used
        dim_k = min(dim_m, dim_n)

        # QR decomposition
        # R = np.linalg.qr(A_ri, 'r')

        # direct QR of stacked matrices for linalg.qr() only works with numpy>=1.22.0
        # workaround for old numpy:
        R = np.empty((n_responses, dim_k, dim_n))
        for i in range(n_responses):
            R[i] = np.linalg.qr(A_ri[i], mode='r')

        # only R22 is required to solve for c_res and d_res
        # R12 and R22 can have a different number of rows, depending on K
        if dim_k == dim_m:
            # K = M
            n_rows_r12 = n_freqs
            n_rows_r22 = n_freqs
        else:
            # K = N
            n_rows_r12 = n_cols_unused
            n_rows_r22 = n_cols_used
        R22 = R[:, n_rows_r12:, n_cols_unused:]

        # weighting
        R22 = weights_responses[:, None, None] * R22

        # assemble compressed coefficient matrix A_fast by row-stacking individual upper triangular matrices R22
        dim0 = n_responses * n_rows_r22 + 1

        A_fast = np.empty((dim0, n_cols_used))
        A_fast[:-1, :] = R22.reshape((dim0 - 1, n_cols_used))

        # extra equation to avoid trivial solution
        A_fast[-1, idx_res_real] = np.sum(coeff_real.real, axis=0)
        A_fast[-1, idx_res_complex_re] = np.sum(coeff_complex_re.real, axis=0)
        A_fast[-1, idx_res_complex_im] = np.sum(coeff_complex_im.real, axis=0)
        A_fast[-1, -1] = n_freqs

        # weighting
        A_fast[-1, :] = weight_extra * A_fast[-1, :]

        scaling = 1 / np.linalg.norm(A_fast, axis=0)
        A_fast = scaling * A_fast

        # right hand side vector (weighted)
        b = np.zeros(dim0)
        b[-1] = weight_extra * n_samples

        # check condition of the linear system
        cond = np.linalg.cond(A_fast)
        full_rank = np.min(A_fast.shape)

        # solve least squares for real parts
        x, residuals, rank, singular_vals = np.linalg.lstsq(A_fast, b, rcond=None)

        x = scaling * x

        # rank deficiency
        rank_deficiency = full_rank - rank

        # assemble individual result vectors from single LS result x
        c_res = x[:-1]
        d_res = x[-1]

        # check if d_res is suited for zeros calculation
        tol_res = 1e-8
        if np.abs(d_res) < tol_res:
            # d_res is too small, discard solution and proceed the |d_res| = tol_res
            logger.info(f'Replacing d_res solution as it was too small ({d_res}).')
            d_res = tol_res * (d_res / np.abs(d_res))

        # build test matrix H, which will hold the new poles as eigenvalues
        H = np.zeros((len(c_res), len(c_res)))

        poles_real = poles[np.nonzero(real_mask)]
        poles_cplx = poles[np.nonzero(~real_mask)]

        H[idx_res_real, idx_res_real] = poles_real.real
        H[idx_res_real] -= c_res / d_res

        H[idx_res_complex_re, idx_res_complex_re] = poles_cplx.real
        H[idx_res_complex_re, idx_res_complex_im] = poles_cplx.imag
        H[idx_res_complex_im, idx_res_complex_re] = -1 * poles_cplx.imag
        H[idx_res_complex_im, idx_res_complex_im] = poles_cplx.real
        H[idx_res_complex_re] -= 2 * c_res / d_res

        poles_new = np.linalg.eigvals(H)

        # replace poles for next iteration
        # complex poles need to come in complex conjugate pairs; append only the positive part
        poles = poles_new[np.nonzero(poles_new.imag >= 0)]

        # flip real part of unstable poles (real part needs to be negative for stability)
        poles.real = -1 * np.abs(poles.real)

        return poles, d_res, cond, rank_deficiency, residuals, singular_vals

    @staticmethod
    def _fit_residues(poles, freqs, freq_responses, fit_constant, fit_proportional, enforce_dc=True):
        n_responses, n_freqs = np.shape(freq_responses)
        omega = 2 * np.pi * freqs
        s = 1j * omega

        # We need two columns for complex poles and one column for real poles in A matrix.
        # This number equals the model order.
        n_cols = VectorFitting.get_model_order(poles)

        idx_constant = []
        idx_proportional = []
        if fit_constant:
            idx_constant = [n_cols]
            n_cols += 1
        if fit_proportional:
            idx_proportional = [n_cols]
            n_cols += 1

        # list of indices in 'poles' with real and with complex values
        real_mask = poles.imag == 0
        idx_poles_real = np.nonzero(real_mask)[0]
        idx_poles_complex = np.nonzero(~real_mask)[0]

        # find and save indices of real and complex poles in the poles list
        i = 0
        idx_res_real = []
        idx_res_complex_re = []
        idx_res_complex_im = []
        for pole in poles:
            if pole.imag == 0:
                idx_res_real.append(i)
                i += 1
            else:
                idx_res_complex_re.append(i)
                idx_res_complex_im.append(i + 1)
                i += 2

        # complex coefficient matrix of shape [N_freqs, n_cols]
        # layout of each row:
        # [pole1, pole2, ..., (constant), (proportional)]
        A = np.empty((n_freqs, n_cols), dtype=complex)

        # calculate coefficients for real and complex residues in the solution vector
        #
        # real pole-residue term (r = r', p = p'):
        # fractional term is r' / (s - p')
        # coefficient for r' is 1 / (s - p')
        coeff_real = 1 / (s[:, None] - poles[None, idx_poles_real])

        # complex-conjugate pole-residue pair (r = r' + j r'', p = p' + j p''):
        # fractional term is r / (s - p) + conj(r) / (s - conj(p))
        #                   = [1 / (s - p) + 1 / (s - conj(p))] * r' + [1j / (s - p) - 1j / (s - conj(p))] * r''
        # coefficient for r' is 1 / (s - p) + 1 / (s - conj(p))
        # coefficient for r'' is 1j / (s - p) - 1j / (s - conj(p))
        coeff_complex_re = (1 / (s[:, None] - poles[None, idx_poles_complex]) +
                            1 / (s[:, None] - np.conj(poles[None, idx_poles_complex])))
        coeff_complex_im = (1j / (s[:, None] - poles[None, idx_poles_complex]) -
                            1j / (s[:, None] - np.conj(poles[None, idx_poles_complex])))

        # part 1: first sum of rational functions (variable c)
        A[:, idx_res_real] = coeff_real
        A[:, idx_res_complex_re] = coeff_complex_re
        A[:, idx_res_complex_im] = coeff_complex_im

        # part 2: constant (variable d) and proportional term (variable e)
        A[:, idx_constant] = 1
        A[:, idx_proportional] = s[:, None]

        scaling = 1 / np.linalg.norm(A, axis=0)
        A = scaling * A

        # DC POINT ENFORCEMENT
        if enforce_dc and freqs[0] == 0.0:
            # data contains the dc point; enforce dc point via linear equality constraint:
            # 1: remove one variable from the solution vector (constant term, if possible).
            # 2: solve remaining linear system (without data at dc) with regular least-squares, as usual. the size of
            #    the solution vector, the coefficient matrix, and the right-hand side are reduced by 1
            # 3: calculate the removed variable (constant term) with the data from the dc point
            #
            # linear system: A * x = b
            # solution vector x contains the unknown residues
            # right-hand side b contains the frequency response to be fitted, sorted by ascending frequency (dc first)
            # coefficient matrix A and vector b are split: A = [[A11, A12], [A21, A22]], b = [[b1], [b2]]
            # [A11, A12] is the first row used later for dc enforcement
            # A21 is a column vector, which is not required anymore
            # A22 is the rest of the matrix for usual least-squares fitting

            # indexing mask of constrained variable in the columns of matrix A
            mask_idx_constrained = np.zeros(n_cols, dtype=bool)
            if fit_constant:
                # use constant term for constrained
                mask_idx_constrained[idx_constant] = True
            else:
                # constant term not present; arbitrarily use first residue instead
                mask_idx_constrained[0] = True

            A22 = A[1:, ~mask_idx_constrained]
            b2 = freq_responses[:, 1:]

            A22_ri = np.vstack((A22.real, A22.imag))
            b22_ri = np.hstack((b2.real, b2.imag))

            logger.info(f'Condition number of coefficient matrix = {int(np.linalg.cond(A22_ri))}')

            # solve least-squares and obtain results as stack of real part vector and imaginary part vector
            x2, residuals, rank, singular_vals = np.linalg.lstsq(A22_ri, b22_ri.T, rcond=None)

            # solve for x1 using the first row (the dc row):
            b1 = freq_responses[:, 0]
            A11 = A[0, mask_idx_constrained]
            A12 = A[0, ~mask_idx_constrained]
            x1 = np.real(1 / A11 * (b1 - np.dot(A12, x2)))

            # reassemble x from x1 and x2
            x = np.empty((n_cols, n_responses))
            x[mask_idx_constrained, :] = x1
            x[~mask_idx_constrained, :] = x2
        else:
            # dc point not included; use and solve the entire linear system with least-squares
            A_ri = np.vstack((A.real, A.imag))
            b_ri = np.hstack((freq_responses.real, freq_responses.imag))

            logger.info(f'Condition number of coefficient matrix = {int(np.linalg.cond(A_ri))}')

            # solve least-squares and obtain results as stack of real part vector and imaginary part vector
            x, residuals, rank, singular_vals = np.linalg.lstsq(A_ri, b_ri.T, rcond=None)

        x = scaling[:, None] * x

        # extract residues from solution vector and align them with poles to get matching pole-residue pairs
        residues = np.empty((len(freq_responses), len(poles)), dtype=complex)
        residues[:, idx_poles_real] = np.transpose(x[idx_res_real])
        residues[:, idx_poles_complex] = np.transpose(x[idx_res_complex_re] + 1j * x[idx_res_complex_im])

        # extract constant and proportional coefficient, if available
        if fit_constant:
            constant_coeff = x[idx_constant][0]
        else:
            constant_coeff = np.zeros(n_responses)

        if fit_proportional:
            proportional_coeff = x[idx_proportional][0]
        else:
            proportional_coeff = np.zeros(n_responses)

        return residues, constant_coeff, proportional_coeff, residuals, rank, singular_vals

    @staticmethod
    def _get_delta(poles, residues, constant_coeff, proportional_coeff, freqs, freq_responses, weights_responses):
        s = 2j * np.pi * freqs
        model = proportional_coeff[:, None] * s + constant_coeff[:, None]
        for i, pole in enumerate(poles):
            if np.imag(pole) == 0.0:
                # real pole
                model += residues[:, i, None] / (s - pole)
            else:
                # complex conjugate pole
                model += (residues[:, i, None] / (s - pole) +
                          np.conjugate(residues[:, i, None]) / (s - np.conjugate(pole)))

        # compute weighted error and return global maximum at each frequency across all individual responses
        delta = np.abs(model - freq_responses) * weights_responses[:, None]

        return np.max(delta, axis=0)

    @staticmethod
    def _find_error_bands(freqs, delta):
        # compute error bands (maximal fit deviation)
        delta_mean = np.mean(delta)
        error = delta - delta_mean

        # find limits of error bands
        idx_limits = np.nonzero(np.diff(error > 0))[0]
        idx_limits_filtered = idx_limits[np.diff(idx_limits, prepend=0) > 2]

        freqs_bands = np.split(freqs, idx_limits_filtered)
        error_bands = np.split(error, idx_limits_filtered)
        n_bands = len(freqs_bands)

        idx_freqs_start = []
        idx_freqs_stop = []
        idx_freqs_max = []
        delta_mean_bands = []
        for i_band in range(n_bands):
            band_error_mean = np.mean(error_bands[i_band])
            if band_error_mean > 0:
                # band with excess error;
                # find frequency index of error maximum inside this band
                i_band_max_error = np.argmax(error_bands[i_band])
                i_start = np.nonzero(freqs == freqs_bands[i_band][0])[0][0]
                i_stop = np.nonzero(freqs == freqs_bands[i_band][-1])[0][0]
                i_max = np.nonzero(freqs == freqs_bands[i_band][i_band_max_error])[0][0]
                idx_freqs_start.append(i_start)
                idx_freqs_stop.append(i_stop)
                idx_freqs_max.append(i_max)
                delta_mean_bands.append(np.mean(delta[i_start:i_stop]))

        idx_freqs_start = np.array(idx_freqs_start)
        idx_freqs_stop = np.array(idx_freqs_stop)
        idx_freqs_max = np.array(idx_freqs_max)
        delta_mean_bands = np.array(delta_mean_bands)

        i_sort = np.flip(np.argsort(delta_mean_bands))

        return idx_freqs_start[i_sort], idx_freqs_stop[i_sort], idx_freqs_max[i_sort], delta_mean_bands[i_sort]

    def get_rms_error(self, i=-1, j=-1, parameter_type: str = 's'):
        r"""
        Returns the root-mean-square (rms) error magnitude of the fit, i.e.
        :math:`\sqrt{ \mathrm{mean}(|S - S_\mathrm{fit} |^2) }`,
        either for an individual response :math:`S_{i+1,j+1}` or for larger slices of the network.

        Parameters
        ----------
        i : int, optional
            Row indices of the responses to be evaluated. Either a single row selected by an integer
            :math:`i \in [0, N_\mathrm{ports}-1]`, or multiple rows selected by a list of integers, or all rows
            selected by :math:`i = -1` (*default*).

        j : int, optional
            Column indices of the responses to be evaluated. Either a single column selected by an integer
            :math:`j \in [0, N_\mathrm{ports}-1]`, or multiple columns selected by a list of integers, or all columns
            selected by :math:`j = -1` (*default*).

        parameter_type: str, optional
            Representation type of the fitted frequency responses. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`).

        Returns
        -------
        rms_error : ndarray
            The rms error magnitude between the vector fitted model and the original network data.

        Raises
        ------
        ValueError
            If the specified parameter representation type is not :attr:`s`, :attr:`z`, nor :attr:`y`.
        """

        if i == -1:
            list_i = range(self.network.nports)
        elif isinstance(i, int):
            list_i = [i]
        else:
            list_i = i

        if j == -1:
            list_j = range(self.network.nports)
        elif isinstance(j, int):
            list_j = [j]
        else:
            list_j = j

        if parameter_type.lower() == 's':
            nw_responses = self.network.s
        elif parameter_type.lower() == 'z':
            nw_responses = self.network.z
        elif parameter_type.lower() == 'y':
            nw_responses = self.network.y
        else:
            raise ValueError(f'Invalid parameter type `{parameter_type}`. Valid options: `s`, `z`, or `y`')

        error_mean_squared = 0
        for i in list_i:
            for j in list_j:
                nw_ij = nw_responses[:, i, j]
                fit_ij = self.get_model_response(i, j, self.network.f)
                error_mean_squared += np.mean(np.square(np.abs(nw_ij - fit_ij)))

        return np.sqrt(error_mean_squared)

    def _get_ABCDE(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Private method.
        Returns the real-valued system matrices of the state-space representation of the current rational model, as
        defined in [#]_.

        Returns
        -------
        A : ndarray
            State-space matrix A holding the poles on the diagonal as real values with imaginary parts on the sub-
            diagonal
        B : ndarray
            State-space matrix B holding coefficients (1, 2, or 0), depending on the respective type of pole in A
        C : ndarray
            State-space matrix C holding the residues
        D : ndarray
            State-space matrix D holding the constants
        E : ndarray
            State-space matrix E holding the proportional coefficients (usually 0 in case of fitted S-parameters)

        Raises
        ------
        ValueError
            If the model parameters have not been initialized (by running :func:`vector_fit()` or :func:`read_npz()`).

        References
        ----------
        .. [#] B. Gustavsen and A. Semlyen, "Fast Passivity Assessment for S-Parameter Rational Models Via a Half-Size
            Test Matrix," in IEEE Transactions on Microwave Theory and Techniques, vol. 56, no. 12, pp. 2701-2708,
            Dec. 2008, DOI: 10.1109/TMTT.2008.2007319.
        """

        # initial checks
        if self.poles is None:
            raise ValueError('self.poles = None; nothing to do. You need to run vector_fit() first.')
        if self.residues is None:
            raise ValueError('self.residues = None; nothing to do. You need to run vector_fit() first.')
        if self.proportional_coeff is None:
            raise ValueError('self.proportional_coeff = None; nothing to do. You need to run vector_fit() first.')
        if self.constant_coeff is None:
            raise ValueError('self.constant_coeff = None; nothing to do. You need to run vector_fit() first.')

        # assemble real-valued state-space matrices A, B, C, D, E from fitted complex-valued pole-residue model

        # determine size of the matrix system
        n_ports = int(np.sqrt(len(self.constant_coeff)))
        n_poles_real = 0
        n_poles_cplx = 0
        for pole in self.poles:
            if np.imag(pole) == 0.0:
                n_poles_real += 1
            else:
                n_poles_cplx += 1
        n_matrix = (n_poles_real + 2 * n_poles_cplx) * n_ports

        # state-space matrix A holds the poles on the diagonal as real values with imaginary parts on the sub-diagonal
        # state-space matrix B holds coefficients (1, 2, or 0), depending on the respective type of pole in A
        # assemble A = [[poles_real,   0,                  0],
        #               [0,            real(poles_cplx),   imag(poles_cplx],
        #               [0,            -imag(poles_cplx),  real(poles_cplx]]
        A = np.identity(n_matrix)
        B = np.zeros(shape=(n_matrix, n_ports))
        i_A = 0  # index on diagonal of A
        for j in range(n_ports):
            for pole in self.poles:
                if np.imag(pole) == 0.0:
                    # adding a real pole
                    A[i_A, i_A] = np.real(pole)
                    B[i_A, j] = 1
                    i_A += 1
                else:
                    # adding a complex-conjugate pole
                    A[i_A, i_A] = np.real(pole)
                    A[i_A, i_A + 1] = np.imag(pole)
                    A[i_A + 1, i_A] = -1 * np.imag(pole)
                    A[i_A + 1, i_A + 1] = np.real(pole)
                    B[i_A, j] = 2
                    i_A += 2

        # state-space matrix C holds the residues
        # assemble C = [[R1.11, R1.12, R1.13, ...], [R2.11, R2.12, R2.13, ...], ...]
        C = np.zeros(shape=(n_ports, n_matrix))
        for i in range(n_ports):
            for j in range(n_ports):
                # i: row index
                # j: column index
                i_response = i * n_ports + j

                j_residues = 0
                for zero in self.residues[i_response]:
                    if np.imag(zero) == 0.0:
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_residues] = np.real(zero)
                        j_residues += 1
                    else:
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_residues] = np.real(zero)
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_residues + 1] = np.imag(zero)
                        j_residues += 2

        # state-space matrix D holds the constants
        # assemble D = [[d11, d12, ...], [d21, d22, ...], ...]
        D = np.zeros(shape=(n_ports, n_ports))
        for i in range(n_ports):
            for j in range(n_ports):
                # i: row index
                # j: column index
                i_response = i * n_ports + j
                D[i, j] = self.constant_coeff[i_response]

        # state-space matrix E holds the proportional coefficients (usually 0 in case of fitted S-parameters)
        # assemble E = [[e11, e12, ...], [e21, e22, ...], ...]
        E = np.zeros(shape=(n_ports, n_ports))
        for i in range(n_ports):
            for j in range(n_ports):
                # i: row index
                # j: column index
                i_response = i * n_ports + j
                E[i, j] = self.proportional_coeff[i_response]

        return A, B, C, D, E

    @staticmethod
    def _get_s_from_ABCDE(freqs: np.ndarray,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Private method.
        Returns the S-matrix of the vector fitted model calculated from the real-valued system matrices of the state-
        space representation, as provided by `_get_ABCDE()`.

        Parameters
        ----------
        freqs : ndarray
            Frequencies (in Hz) at which to calculate the S-matrices.
        A : ndarray
        B : ndarray
        C : ndarray
        D : ndarray
        E : ndarray

        Returns
        -------
        ndarray
            Complex-valued S-matrices (fxNxN) calculated at frequencies `freqs`.
        """

        dim_A = np.shape(A)[0]
        stsp_poles = np.linalg.inv(2j * np.pi * freqs[:, None, None] * np.identity(dim_A)[None, :, :] - A[None, :, :])
        stsp_S = np.matmul(np.matmul(C, stsp_poles), B)
        stsp_S += D + 2j * np.pi * freqs[:, None, None] * E
        return stsp_S

    def passivity_test(self, parameter_type: str = 's') -> np.ndarray:
        """
        Evaluates the passivity of reciprocal vector fitted models by means of a half-size test matrix [#]_. Any
        existing frequency bands of passivity violations will be returned as a sorted list.

        Parameters
        ----------
        parameter_type: str, optional
            Representation type of the fitted frequency responses. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`). Currently, only scattering
            parameters are supported for passivity evaluation.

        Raises
        ------
        NotImplementedError
            If the function is called for `parameter_type` different than `S` (scattering).

        ValueError
            If the function is used with a model containing nonzero proportional coefficients.

        Returns
        -------
        violation_bands : ndarray
            NumPy array with frequency bands of passivity violation:
            `[[f_start_1, f_stop_1], [f_start_2, f_stop_2], ...]`.

        See Also
        --------
        is_passive : Query the model passivity as a boolean value.
        passivity_enforce : Enforces the passivity of the vector fitted model, if required.

        Examples
        --------
        Load and fit the `Network`, then evaluate the model passivity:

        >>> nw_3port = skrf.Network('my3port.s3p')
        >>> vf = skrf.VectorFitting(nw_3port)
        >>> vf.vector_fit(n_poles_real=1, n_poles_cmplx=4)
        >>> violations = vf.passivity_test()

        References
        ----------
        .. [#] B. Gustavsen and A. Semlyen, "Fast Passivity Assessment for S-Parameter Rational Models Via a Half-Size
            Test Matrix," in IEEE Transactions on Microwave Theory and Techniques, vol. 56, no. 12, pp. 2701-2708,
            Dec. 2008, DOI: 10.1109/TMTT.2008.2007319.
        """

        if parameter_type.lower() != 's':
            raise NotImplementedError('Passivity testing is currently only supported for scattering (S) parameters.')
        if parameter_type.lower() == 's' and len(np.flatnonzero(self.proportional_coeff)) > 0:
            raise ValueError('Passivity testing of scattering parameters with nonzero proportional coefficients does '
                             'not make any sense; you need to run vector_fit() with option `fit_proportional=False` '
                             'first.')

        # # the network needs to be reciprocal for this passivity test method to work: S = transpose(S)
        # if not np.allclose(self.residues, np.transpose(self.residues)) or \
        #         not np.allclose(self.constant_coeff, np.transpose(self.constant_coeff)) or \
        #         not np.allclose(self.proportional_coeff, np.transpose(self.proportional_coeff)):
        #     logger.error('Passivity testing with unsymmetrical model parameters is not supported. '
        #                   'The model needs to be reciprocal.')
        #     return

        # get state-space matrices
        A, B, C, D, E = self._get_ABCDE()
        n_ports = np.shape(D)[0]

        # build half-size test matrix P from state-space matrices A, B, C, D
        inv_neg = np.linalg.inv(D - np.identity(n_ports))
        inv_pos = np.linalg.inv(D + np.identity(n_ports))
        prod_neg = np.matmul(np.matmul(B, inv_neg), C)
        prod_pos = np.matmul(np.matmul(B, inv_pos), C)
        P = np.matmul(A - prod_neg, A - prod_pos)

        # extract eigenvalues of P
        P_eigs = np.linalg.eigvals(P)

        # purely imaginary square roots of eigenvalues identify frequencies (2*pi*f) of borders of passivity violations
        freqs_violation = []
        for sqrt_eigenval in np.sqrt(P_eigs):
            if np.real(sqrt_eigenval) == 0.0:
                freqs_violation.append(np.imag(sqrt_eigenval) / 2 / np.pi)

        # include dc (0) unless it's already included
        if len(np.nonzero(np.array(freqs_violation) == 0.0)[0]) == 0:
            freqs_violation.append(0.0)

        # sort the output from lower to higher frequencies
        freqs_violation = np.sort(freqs_violation)

        # identify frequency bands of passivity violations

        # sweep the bands between crossover frequencies and identify bands of passivity violations
        violation_bands = []
        for i, freq in enumerate(freqs_violation):
            if i == len(freqs_violation) - 1:
                # last band stops always at infinity
                f_start = freq
                f_stop = np.inf
                f_center = 1.1 * f_start # 1.1 is chosen arbitrarily to have any frequency for evaluation
            else:
                # intermediate band between this frequency and the previous one
                f_start = freq
                f_stop = freqs_violation[i + 1]
                f_center = 0.5 * (f_start + f_stop)

            # calculate singular values at the center frequency between crossover frequencies to identify violations
            s_center = self._get_s_from_ABCDE(np.array([f_center]), A, B, C, D, E)
            sigma = np.linalg.svd(s_center[0], compute_uv=False)
            passive = True
            for singval in sigma:
                if singval > 1:
                    # passivity violation in this band
                    passive = False
            if not passive:
                # add this band to the list of passivity violations
                if violation_bands is None:
                    violation_bands = [[f_start, f_stop]]
                else:
                    violation_bands.append([f_start, f_stop])

        return np.array(violation_bands)

    def is_passive(self, parameter_type: str = 's') -> bool:
        """
        Returns the passivity status of the model as a boolean value.

        Parameters
        ----------
        parameter_type : str, optional
            Representation type of the fitted frequency responses. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`). Currently, only scattering
            parameters are supported for passivity evaluation.

        Returns
        -------
        passivity : bool
            :attr:`True` if model is passive, else :attr:`False`.

        See Also
        --------
        passivity_test : Verbose passivity evaluation routine.
        passivity_enforce : Enforces the passivity of the vector fitted model, if required.

        Examples
        --------
        Load and fit the `Network`, then check whether or not the model is passive:

        >>> nw_3port = skrf.Network('my3port.s3p')
        >>> vf = skrf.VectorFitting(nw_3port)
        >>> vf.vector_fit(n_poles_real=1, n_poles_cmplx=4)
        >>> vf.is_passive() # returns True or False
        """

        viol_bands = self.passivity_test(parameter_type)
        if len(viol_bands) == 0:
            return True
        else:
            return False

    def passivity_enforce(self, n_samples: int = 200, f_max: float = None, parameter_type: str = 's',
                          preserve_dc: bool = True) -> None:
        """
        Enforces the passivity of the vector fitted model, if required. This is an implementation of the methods
        presented in [#]_ and [#]_ using singular value perturbation. To preserve the dc point in the model during
        passivity enforcement, only the residues are perturbed, not the constant term.

        Parameters
        ----------
        n_samples : int, optional
            Number of linearly spaced frequency samples at which passivity will be evaluated and enforced.
            (Default: 200). If there are very narrow frequency bands of passivity violations, a sufficiently large
            number of frequency samples is required.

        f_max : float or None, optional
            Highest frequency of interest for the passivity enforcement (in Hz, not rad/s). This limit usually
            equals the highest sample frequency of the fitted Network. If None, the highest frequency in
            :attr:`self.network` is used, which must not be None is this case. If `f_max` is not None, it overrides the
            highest frequency in :attr:`self.network`.

        parameter_type : str, optional
            Representation type of the fitted frequency responses. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`). Currently, only scattering
            parameters are supported for passivity evaluation.

        preserve_dc : bool, optional
            Enables dc point preservation during passivity enforcement. This only works if the fitted model is already
            passive at the dc point, which is not always the case. If it is not passive, dc point preservation is
            disabled and passivity is also enforced on the dc point.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the function is called for `parameter_type` different than `S` (scattering).

        ValueError
            If the function is used with a model containing nonzero proportional coefficients. Or if both `f_max` and
            :attr:`self.network` are None.

        See Also
        --------
        is_passive : Returns the passivity status of the model as a boolean value.
        passivity_test : Verbose passivity evaluation routine.
        plot_passivation : Convergence plot for passivity enforcement iterations.

        Examples
        --------
        Load and fit the `Network`, then enforce the passivity of the model:

        >>> nw_3port = skrf.Network('my3port.s3p')
        >>> vf = skrf.VectorFitting(nw_3port)
        >>> vf.vector_fit(n_poles_real=1, n_poles_cmplx=4)
        >>> vf.passivity_enforce()  # won't do anything if model is already passive

        References
        ----------
        .. [#] T. Dhaene, D. Deschrijver and N. Stevens, "Efficient Algorithm for Passivity Enforcement of S-Parameter-
            Based Macromodels," in IEEE Transactions on Microwave Theory and Techniques, vol. 57, no. 2, pp. 415-420,
            Feb. 2009, DOI: 10.1109/TMTT.2008.2011201

        .. [#] D. Deschrijver and T. Dhaene, "DC-Preserving Passivity Enforcement for S-Parameter Based Macromodels,"
            in IEEE Transactions on Microwave Theory and Techniques, vol. 58, no. 4, pp. 923-928, April 2010,
            DOI: 10.1109/TMTT.2010.2042556
        """

        if parameter_type.lower() != 's':
            raise NotImplementedError('Passivity testing is currently only supported for scattering (S) parameters.')
        if parameter_type.lower() == 's' and len(np.flatnonzero(self.proportional_coeff)) > 0:
            raise ValueError('Passivity testing of scattering parameters with nonzero proportional coefficients does '
                             'not make any sense; you need to run vector_fit() with option `fit_proportional=False` '
                             'first.')

        # always run passivity test first; this will write 'self.violation_bands'
        if self.is_passive():
            # model is already passive; do nothing and return
            logger.info('Passivity enforcement: The model is already passive. Nothing to do.')
            return

        # check dc passivity and find the highest relevant frequency; either
        # 1) the highest frequency of passivity violation (f_viol_max)
        # or
        # 2) the highest fitting frequency (f_samples_max)
        violation_bands = self.passivity_test()
        f_viol_min = violation_bands[0, 0]
        f_viol_max = violation_bands[-1, 1]

        # check passivity at the dc point; 1) in the model, 2) in the original data, if available
        if preserve_dc and f_viol_min == 0.0:
            # cannot preserve a non-passive dc point during passivity enforcement
            preserve_dc = False
            hint = ''

            if self.network is not None:
                if self.network.f[0] == 0.0 and not self.network.is_passive():
                    hint = '\nHint: The dc point in the original network data is already non-passive.'

            warnings.warn('Passivity enforcement: The dc point in the model is not passive. Cannot '
                          f'preserve the dc point during passivity enforcement. {hint}', UserWarning, stacklevel=2)

        if f_max is None:
            if self.network is None:
                raise RuntimeError('Both `self.network` and parameter `f_max` are None. One of them is required to '
                                   'specify the frequency band of interest for the passivity enforcement.')
            else:
                f_samples_max = self.network.f[-1]
        else:
            f_samples_max = f_max

        # deal with unbounded violation interval (f_viol_max == np.inf)
        if np.isinf(f_viol_max):
            f_viol_max = 1.5 * violation_bands[-1, 0]
            warnings.warn(
                'Passivity enforcement: The passivity violations of this model are unbounded. '
                'Passivity enforcement might still work, but consider re-fitting with a lower number of poles '
                'and/or without the constants (`fit_constant=False`) if the results are not satisfactory.',
                UserWarning, stacklevel=2)

        # the frequency band for the passivity evaluation is from dc to 20% above the highest relevant frequency
        if f_viol_max < f_samples_max:
            f_eval_max = 1.2 * f_samples_max
        else:
            f_eval_max = 1.2 * f_viol_max

        # let's not automatically adjust n_samples. The calculated number can
        # be huge (>100k). Combined with a high number of poles in the model, this can bust the memory.
        freqs_eval = np.linspace(0, f_eval_max, n_samples)

        # get model state-space matrices
        A, B, C_t, D, E = self._get_ABCDE()
        dim_A = np.shape(A)[0]

        # ASYMPTOTIC PASSIVITY ENFORCEMENT

        # check if constant term has been fitted (not zero)
        # a model without the constant term is always asymptotically passive
        if len(np.nonzero(D)[0]) != 0:
            # D was fitted;
            # asymptotic passivity needs to be checked and enforced, if violated.
            # for dc preservation, the asymptotic passivity violations in D are compensated using C
            # D is not touched, because it contains the dc point ( lim s --> {inf S(s)} = D)
            u, sigma, vh = np.linalg.svd(D, compute_uv=True)

            # find and perturb singular values that cause passivity violations
            # sigma_viol = sigma * upsilon - psi with
            #       upsilon[sigma > delta] = 1
            #       upsilon[sigma <= delta] = 0
            #       psi[sigma > delta] = delta
            #       psi[sigma <= delta] = 0
            # (implemented below in a more compact form)
            delta = 1
            idx_viol = np.nonzero(sigma > delta)
            sigma_viol = np.zeros_like(sigma)
            sigma_viol[idx_viol] = sigma[idx_viol] - delta

            # calculate S_viol from perturbed sigma and previous U and Vh
            S_viol = np.dot(u * sigma_viol, vh)

            # find new set of residues C_viol by solving underdetermined least-squares problem
            # S_viol = C_viol * B
            #
            # mind the transpose of the system to compensate for the exchanged order of matrix multiplication:
            # S_viol = C_viol * B <==> transpose(S_viol) = transpose(B) * transpose(C_viol)
            C_viol, residuals, rank, singular_vals = np.linalg.lstsq(np.vstack((B.T.real, B.T.imag)),
                                                                     np.vstack((S_viol.T.real, S_viol.T.imag)),
                                                                     rcond=None)
            C_t -= C_viol.T

        # UNIFORM PASSIVITY ENFORCEMENT

        # preparing coefficient matrix; can be reused in every iteration
        # S(s_eval) = D_t + s_eval * C_t * inv(s_eval * I - A) * B
        #           = D_t + s_eval * C_t * A_freq * B
        # with
        #   A_freq = inv(s_eval * I - A)
        #   s_eval = j * omega_eval = 2j * pi * freqs_eval
        A_freq = np.linalg.inv(2j * np.pi * freqs_eval[:, None, None] * np.identity(dim_A)[None, :, :] - A[None, :, :])

        # construct coefficient matrix for least-squares residue fitting (C_viol)
        coeffs = np.matmul(A_freq, B)

        C_viol = np.empty_like(C_t)
        n_ports = np.shape(C_viol)[0]
        model_order = self.get_model_order(self.poles)

        # predefined tolerance parameter (users should not need to change this)
        delta_threshold = 0.999
        sigma_max = 1.1     # just to enter iteration loop for the first time

        # iterative compensation of passivity violations
        t = 0
        self.history_max_sigma = []
        while t < self.max_iterations and sigma_max > 1.0:
            logger.info(f'Passivity enforcement; Iteration {t + 1}')

            # calculate S-matrix of the model at freqs_eval (shape fxNxN)
            #S_eval = self._get_s_from_ABCDE(freqs_eval, A, B, C_t, D, E)
            S_eval = D + np.matmul(C_t, coeffs)   # much faster!

            # singular value decomposition,
            # shape(u) = (n_samples, n_ports, n_ports)
            # shape(sigma) = (n_samples, n_ports)
            # shape(vh) = (n_samples, n_ports, n_ports)
            u, sigma, vh = np.linalg.svd(S_eval)

            # keep track of the greatest singular value in every iteration step
            sigma_max = np.amax(sigma)
            self.history_max_sigma.append(sigma_max)

            if sigma_max > delta_threshold:
                delta = delta_threshold
            else:
                delta = sigma_max

            # find and perturb singular values that cause passivity violations
            # sigma_viol = sigma * upsilon - psi with
            #       upsilon[sigma > delta] = 1
            #       upsilon[sigma <= delta] = 0
            #       psi[sigma > delta] = delta
            #       psi[sigma <= delta] = 0
            # (implemented below in a more compact form)
            idx_viol = np.nonzero(sigma > delta)
            sigma_viol = np.zeros_like(sigma)
            sigma_viol[idx_viol] = sigma[idx_viol] - delta

            S_viol = np.matmul(u * sigma_viol[:, None, :], vh)

            # stack frequency responses as a single vector
            # stacking order (row-major):
            # s11, s12, s13, ..., s21, s22, s23, ...
            S_viol_stacked = []
            for i in range(n_ports):
                for j in range(n_ports):
                    S_viol_stacked.append(S_viol[:, i, j])
            S_viol_stacked = np.array(S_viol_stacked)

            # The existing method _fit_residues() can be use here to fit the violation residues. Enabling `fit_constant`
            # in combination with `enforce_dc` removes the dc rows from the linear system and enforces the dc solution
            # on the constant term. In case of dc preservation during passivity enforcement, we can ignore that constant
            # term entirely and only use the violation residues.
            # If dc preservation is disabled, we could also perturb the constant term. This is not currently done. In
            # this new method, we always only perturb the residues. Disabling `fit_constant` and `preserve_dc` in this
            # case will solve for the residues without the constant term in the linear system.
            C_viol_stacked, D_viol_stacked, E_viol_stacked, residuals, rank, singular_vals = self._fit_residues(
                self.poles, freqs_eval, S_viol_stacked, fit_constant=preserve_dc, fit_proportional=False,
                enforce_dc=preserve_dc)

            # reshape C_viol into state-space format: [[R1.11, R2.11, R3.11, ..., R1.1N, R2.1N, R3.1N, ...],
            #                                          [R1.21, R2.21, R3.21, ..., R1.2N, R3.2N, R3.2N, ...],
            #                                           ...
            #                                          [R1.N1, R2.N1, R3.N1, ..., R1.NN, R3.NN, R3.NN, ...]]
            for i_port in range(n_ports):
                for j_port in range(n_ports):
                    j_residues = 0
                    for residue in C_viol_stacked[i_port * n_ports + j_port]:
                        if np.imag(residue) == 0.0:
                            C_viol[i_port, j_port * model_order + j_residues] = np.real(residue)
                            j_residues += 1
                        else:
                            C_viol[i_port, j_port * model_order + j_residues] = np.real(residue)
                            C_viol[i_port, j_port * model_order + j_residues + 1] = np.imag(residue)
                            j_residues += 2

            # perturb residues by subtracting respective row and column in C_t
            C_t = C_t - C_viol

            t += 1

        # PASSIVATION PROCESS DONE; model is either passive or max. number of iterations have been exceeded
        if t == self.max_iterations:
            warnings.warn('Passivity enforcement: Aborting after the max. number of iterations has been '
                          'exceeded.', RuntimeWarning, stacklevel=2)

        # save/update model parameters (perturbed residues)
        self.history_max_sigma = np.array(self.history_max_sigma)

        n_ports = np.shape(D)[0]
        for i in range(n_ports):
            k = 0   # column index in C_t
            for j in range(n_ports):
                i_response = i * n_ports + j
                z = 0   # column index self.residues
                for pole in self.poles:
                    if np.imag(pole) == 0.0:
                        # real pole --> real residue
                        self.residues[i_response, z] = C_t[i, k]
                        k += 1
                    else:
                        # complex-conjugate pole --> complex-conjugate residue
                        self.residues[i_response, z] = C_t[i, k] + 1j * C_t[i, k + 1]
                        k += 2
                    z += 1

        # run final passivity test to make sure passivation was successful
        violation_bands = self.passivity_test()
        if len(violation_bands) > 0:
            # trying to determine the required number of evaluation samples based on the bandwidth and separation
            # distance of the violation bands
            violation_band_separation = np.diff(violation_bands.flat)
            min_spacing_nonzero = np.amin(violation_band_separation[violation_band_separation != 0.0])

            # we should need an absolute minimum of 1 sample in each violating frequency band.
            # in practice, the frequency spacing should preferrably be much more dense.
            # let's recommend 2 samples per violation band.
            n_samples_required = int(f_eval_max / min_spacing_nonzero * 2)

            if n_samples_required > n_samples:
                hint = f'Consider trying again with n_samples > {n_samples_required}.'
            else:
                hint = ''

            warnings.warn('Passivity enforcement was not successful.\nModel is still non-passive in these '
                          f'frequency bands: {violation_bands}.\nTry running this routine again with a larger number of'
                          f' samples (parameter `n_samples`). This run was using n_samples = {n_samples}. {hint}',
                          RuntimeWarning, stacklevel=2)

    def write_npz(self, path: str) -> None:
        """
        Writes the model parameters in :attr:`poles`, :attr:`residues`,
        :attr:`proportional_coeff` and :attr:`constant_coeff` to a labeled NumPy .npz file.

        Parameters
        ----------
        path : str
            Target path without filename for the export. The filename will be added automatically based on the network
            name in :attr:`network`

        Returns
        -------
        None

        See Also
        --------
        read_npz : Reads all model parameters from a .npz file

        Examples
        --------
        Load and fit the `Network`, then export the model parameters to a .npz file:

        >>> nw_3port = skrf.Network('my3port.s3p')
        >>> vf = skrf.VectorFitting(nw_3port)
        >>> vf.vector_fit(n_poles_real=1, n_poles_cmplx=4)
        >>> vf.write_npz('./data/')

        The filename depends on the network name stored in `nw_3port.name` and will have the prefix `coefficients_`, for
        example `coefficients_my3port.npz`. The coefficients can then be read using NumPy's load() function:

        >>> coeffs = numpy.load('./data/coefficients_my3port.npz')
        >>> poles = coeffs['poles']
        >>> residues = coeffs['residues']
        >>> prop_coeffs = coeffs['proportionals']
        >>> constants = coeffs['constants']

        Alternatively, the coefficients can be read directly into a new instance of `VectorFitting`, see
        :func:`read_npz`.
        """

        if self.poles is None:
            warnings.warn('Nothing to export; Poles have not been fitted.', RuntimeWarning, stacklevel=2)
            return
        if self.residues is None:
            warnings.warn('Nothing to export; Residues have not been fitted.', RuntimeWarning, stacklevel=2)
            return
        if self.proportional_coeff is None:
            warnings.warn('Nothing to export; Proportional coefficients have not been fitted.', RuntimeWarning,
                          stacklevel=2)
            return
        if self.constant_coeff is None:
            warnings.warn('Nothing to export; Constants have not been fitted.', RuntimeWarning, stacklevel=2)
            return

        filename = self.network.name

        logger.info(f'Exporting results as compressed NumPy array to {path}')
        np.savez_compressed(os.path.join(path, f'coefficients_{filename}'),
                            poles=self.poles, residues=self.residues, proportionals=self.proportional_coeff,
                            constants=self.constant_coeff)

    def read_npz(self, file: str) -> None:
        """
        Reads all model parameters :attr:`poles`, :attr:`residues`, :attr:`proportional_coeff` and
        :attr:`constant_coeff` from a labeled NumPy .npz file.

        Parameters
        ----------
        file : str
            NumPy .npz file containing the parameters. See notes.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the shapes of the coefficient arrays in the provided file are not compatible.

        Notes
        -----
        The .npz file needs to include the model parameters as individual NumPy arrays (ndarray) labeled '*poles*',
        '*residues*', '*proportionals*' and '*constants*'. The shapes of those arrays need to match the network
        properties in :class:`network` (correct number of ports). Preferably, the .npz file was created by
        :func:`write_npz`.

        See Also
        --------
        write_npz : Writes all model parameters to a .npz file

        Examples
        --------
        Create an empty `VectorFitting` instance (with or without the fitted `Network`) and load the model parameters:

        >>> vf = skrf.VectorFitting(None)
        >>> vf.read_npz('./data/coefficients_my3port.npz')

        This can be useful to analyze or process a previous vector fit instead of fitting it again, which sometimes
        takes a long time. For example, the model passivity can be evaluated and enforced:

        >>> vf.passivity_enforce()
        """

        with np.load(file) as data:
            poles = data['poles']

            # legacy support for exported residues
            if 'zeros' in data:
                # old .npz file from deprecated write_npz() with residues called 'zeros'
                residues = data['zeros']
            else:
                # new .npz file from current write_npz()
                residues = data['residues']

            proportional_coeff = data['proportionals']
            constant_coeff = data['constants']

            n_ports = int(np.sqrt(len(constant_coeff)))
            n_resp = n_ports ** 2
            if np.shape(residues)[0] == np.shape(proportional_coeff)[0] == np.shape(constant_coeff)[0] == n_resp:
                self.poles = poles
                self.residues = residues
                self.proportional_coeff = proportional_coeff
                self.constant_coeff = constant_coeff
            else:
                raise ValueError('The shapes of the provided parameters are not compatible. The coefficient file needs '
                                 'to contain NumPy arrays labled `poles`, `residues`, `proportionals`, and '
                                 '`constants`. Their shapes must match the number of network ports and the number of '
                                 'frequencies.')

    def get_model_response(self, i: int, j: int, freqs: Any = None) -> np.ndarray:
        """
        Returns one of the frequency responses :math:`H_{i+1,j+1}` of the fitted model :math:`H`.

        Parameters
        ----------
        i : int
            Row index of the response in the response matrix.

        j : int
            Column index of the response in the response matrix.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        Returns
        -------
        response : ndarray
            Model response :math:`H_{i+1,j+1}` at the frequencies specified in `freqs` (complex-valued Numpy array).

        Examples
        --------
        Get fitted S11 at 101 frequencies from 0 Hz to 10 GHz:

        >>> import skrf
        >>> vf = skrf.VectorFitting(skrf.data.ring_slot)
        >>> vf.vector_fit(3, 0)
        >>> s11_fit = vf.get_model_response(0, 0, numpy.linspace(0, 10e9, 101))
        """

        if self.poles is None:
            warnings.warn('Returning a zero-vector; Poles have not been fitted.',
                          RuntimeWarning, stacklevel=2)
            return np.zeros_like(freqs)
        if self.residues is None:
            warnings.warn('Returning a zero-vector; Residues have not been fitted.',
                          RuntimeWarning, stacklevel=2)
            return np.zeros_like(freqs)
        if self.proportional_coeff is None:
            warnings.warn('Returning a zero-vector; Proportional coefficients have not been fitted.',
                          RuntimeWarning, stacklevel=2)
            return np.zeros_like(freqs)
        if self.constant_coeff is None:
            warnings.warn('Returning a zero-vector; Constants have not been fitted.',
                          RuntimeWarning, stacklevel=2)
            return np.zeros_like(freqs)
        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        s = 2j * np.pi * np.array(freqs)
        n_ports = int(np.sqrt(len(self.constant_coeff)))
        i_response = i * n_ports + j
        residues = self.residues[i_response]

        resp = self.proportional_coeff[i_response] * s + self.constant_coeff[i_response]
        for i, pole in enumerate(self.poles):
            if np.imag(pole) == 0.0:
                # real pole
                resp += residues[i] / (s - pole)
            else:
                # complex conjugate pole
                resp += residues[i] / (s - pole) + np.conjugate(residues[i]) / (s - np.conjugate(pole))
        return resp

    @axes_kwarg
    def plot(self, component: str, i: int = -1, j: int = -1, freqs: Any = None,
             parameter: str = 's', *, ax: Axes = None) -> Axes:
        """
        Plots the specified component of the parameter :math:`H_{i+1,j+1}` in the fit, where :math:`H` is
        either the scattering (:math:`S`), the impedance (:math:`Z`), or the admittance (:math:`H`) response specified
        in `parameter`.

        Parameters
        ----------
        component : str
            The component to be plotted. Must be one of the following items:
            ['db', 'mag', 'deg', 'deg_unwrap', 're', 'im'].
            `db` for magnitude in decibels,
            `mag` for magnitude in linear scale,
            `deg` for phase in degrees (wrapped),
            `deg_unwrap` for phase in degrees (unwrapped/continuous),
            `re` for real part in linear scale,
            `im` for imaginary part in linear scale.

        i : int, optional
            Row index of the response. `-1` to plot all rows.

        j : int, optional
            Column index of the response. `-1` to plot all columns.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used. This only works if :attr:`network` is not `None`.

        parameter : str, optional
            The network representation to be used. This is only relevant for the plot of the original sampled response
            in :attr:`network` that is used for comparison with the fit. Must be one of the following items unless
            :attr:`network` is `None`: ['s', 'z', 'y'] for *scattering* (default), *impedance*, or *admittance*.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Raises
        ------
        ValueError
            If the `freqs` parameter is not specified while the Network in :attr:`network` is `None`.
            Also if `component` and/or `parameter` are not valid.
        """

        components = ['db', 'mag', 'deg', 'deg_unwrap', 're', 'im']
        if component.lower() in components:
            if self.residues is None or self.poles is None:
                raise RuntimeError('Poles and/or residues have not been fitted. Cannot plot the model response.')

            n_ports = int(np.sqrt(np.shape(self.residues)[0]))

            if i == -1:
                list_i = range(n_ports)
            elif isinstance(i, int):
                list_i = [i]
            else:
                list_i = i

            if j == -1:
                list_j = range(n_ports)
            elif isinstance(j, int):
                list_j = [j]
            else:
                list_j = j

            if self.network is not None:
                # plot the original network response at each sample frequency (scatter plot)
                if parameter.lower() == 's':
                    responses = self.network.s
                elif parameter.lower() == 'z':
                    responses = self.network.z
                elif parameter.lower() == 'y':
                    responses = self.network.y
                else:
                    raise ValueError('The network parameter type is not valid, must be `s`, `z`, or `y`, '
                                     f'got `{parameter}`.')

                i_samples = 0
                for i in list_i:
                    for j in list_j:
                        if i_samples == 0:
                            label = 'Samples'
                        else:
                            label = '_nolegend_'
                        i_samples += 1

                        y_vals = None
                        if component.lower() == 'db':
                            y_vals = 20 * np.log10(np.abs(responses[:, i, j]))
                        elif component.lower() == 'mag':
                            y_vals = np.abs(responses[:, i, j])
                        elif component.lower() == 'deg':
                            y_vals = np.rad2deg(np.angle(responses[:, i, j]))
                        elif component.lower() == 'deg_unwrap':
                            y_vals = np.rad2deg(np.unwrap(np.angle(responses[:, i, j])))
                        elif component.lower() == 're':
                            y_vals = np.real(responses[:, i, j])
                        elif component.lower() == 'im':
                            y_vals = np.imag(responses[:, i, j])

                        ax.scatter(self.network.f, y_vals, color='r', label=label)

                if freqs is None:
                    # get frequency array from the network
                    freqs = self.network.f

            if freqs is None:
                raise ValueError(
                    'Neither `freqs` nor `self.network` is specified. Cannot plot model response without any '
                    'frequency information.')

            # plot the fitted responses
            y_label = ''
            i_fit = 0
            for i in list_i:
                for j in list_j:
                    if i_fit == 0:
                        label = 'Fit'
                    else:
                        label = '_nolegend_'
                    i_fit += 1

                    y_model = self.get_model_response(i, j, freqs)
                    y_vals = None
                    if component.lower() == 'db':
                        y_vals = 20 * np.log10(np.abs(y_model))
                        y_label = 'Magnitude (dB)'
                    elif component.lower() == 'mag':
                        y_vals = np.abs(y_model)
                        y_label = 'Magnitude'
                    elif component.lower() == 'deg':
                        y_vals = np.rad2deg(np.angle(y_model))
                        y_label = 'Phase (Degrees)'
                    elif component.lower() == 'deg_unwrap':
                        y_vals = np.rad2deg(np.unwrap(np.angle(y_model)))
                        y_label = 'Phase (Degrees)'
                    elif component.lower() == 're':
                        y_vals = np.real(y_model)
                        y_label = 'Real Part'
                    elif component.lower() == 'im':
                        y_vals = np.imag(y_model)
                        y_label = 'Imaginary Part'

                    ax.plot(freqs, y_vals, color='k', label=label)

            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(y_label)
            ax.legend(loc='best')

            # only print title if a single response is shown
            if i_fit == 1:
                ax.set_title(f'Response i={i}, j={j}')

            return ax
        else:
            raise ValueError(f'The specified component ("{component}") is not valid. Must be in {components}.')

    def plot_s_db(self, *args, **kwargs) -> Axes:
        """
        Plots the magnitude in dB of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('db', *args, **kwargs)``.
        """

        return self.plot('db', *args, **kwargs)

    def plot_s_mag(self, *args, **kwargs) -> Axes:
        """
        Plots the magnitude in linear scale of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('mag', *args, **kwargs)``.
        """

        return self.plot('mag', *args, **kwargs)

    def plot_s_deg(self, *args, **kwargs) -> Axes:
        """
        Plots the phase in degrees of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('deg', *args, **kwargs)``.
        """

        return self.plot('deg', *args, **kwargs)

    def plot_s_deg_unwrap(self, *args, **kwargs) -> Axes:
        """
        Plots the unwrapped phase in degrees of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('deg_unwrap', *args, **kwargs)``.
        """

        return self.plot('deg_unwrap', *args, **kwargs)

    def plot_s_re(self, *args, **kwargs) -> Axes:
        """
        Plots the real part of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('re', *args, **kwargs)``.
        """

        return self.plot('re', *args, **kwargs)

    def plot_s_im(self, *args, **kwargs) -> Axes:
        """
        Plots the imaginary part of the scattering parameter response(s) in the fit.

        Parameters
        ----------
        *args : any, optional
            Additonal arguments to be passed to :func:`plot`.

        **kwargs : dict, optional
            Additonal keyword arguments to be passed to :func:`plot`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Notes
        -----
        This simply calls ``plot('im', *args, **kwargs)``.
        """

        return self.plot('im', *args, **kwargs)

    @axes_kwarg
    def plot_s_singular(self, freqs: Any = None, *, ax: Axes = None) -> Axes:
        """
        Plots the singular values of the vector fitted S-matrix in linear scale.

        Parameters
        ----------
        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used. This only works if :attr:`network` is not `None`.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.

        Raises
        ------
        ValueError
            If the `freqs` parameter is not specified while the Network in :attr:`network` is `None`.
        """

        if freqs is None:
            if self.network is None:
                raise ValueError(
                    'Neither `freqs` nor `self.network` is specified. Cannot plot model response without any '
                    'frequency information.')
            else:
                freqs = self.network.f

        # get system matrices of state-space representation
        A, B, C, D, E = self._get_ABCDE()

        n_ports = np.shape(D)[0]

        # calculate and save singular values for each frequency
        u, sigma, vh = np.linalg.svd(self._get_s_from_ABCDE(freqs, A, B, C, D, E))

        # plot the frequency response of each singular value
        for n in range(n_ports):
            ax.plot(freqs, sigma[:, n], label=fr'$\sigma_{n + 1}$')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(loc='best')
        return ax

    @axes_kwarg
    def plot_convergence(self, ax: Axes = None) -> Axes:
        """
        Plots the history of the model residue parameter **d_res** during the iterative pole relocation process of the
        vector fitting, which should eventually converge to a fixed value. Additionally, the relative change of the
        maximum singular value of the coefficient matrix **A** are plotted, which serve as a convergence indicator.

        Parameters
        ----------
        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        ax.semilogy(np.arange(len(self.delta_max_history)) + 1, self.delta_max_history, color='darkblue')
        ax.set_xlabel('Iteration step')
        ax.set_ylabel('Max. relative change', color='darkblue')
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(self.d_res_history)) + 1, self.d_res_history, color='orangered')
        ax2.set_ylabel('Residue', color='orangered')
        return ax

    @axes_kwarg
    def plot_passivation(self, ax: Axes = None) -> Axes:
        """
        Plots the history of the greatest singular value during the iterative passivity enforcement process, which
        should eventually converge to a value slightly lower than 1.0 or stop after reaching the maximum number of
        iterations specified in the class variable :attr:`max_iterations`.

        Parameters
        ----------
        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        ax.plot(np.arange(len(self.history_max_sigma)) + 1, self.history_max_sigma)
        ax.set_xlabel('Iteration step')
        ax.set_ylabel('Max. singular value')
        return ax

    def write_spice_subcircuit_s(self, file: str, fitted_model_name: str = "s_equivalent",
                                     create_reference_pins: bool = False) -> None:
        """
        Creates an equivalent N-port subcircuit based on its vector fitted scattering (S) parameter responses
        in spice simulator netlist syntax (compatible with LTspice, ngspice, Xyce, ...). The circuit synthesis is based
        on a direct implementation of the state-space representation of the vector fitted model [#vf-book]_.

        Parameters
        ----------
        file : str
            Path and filename including file extension (usually .sp) for the subcircuit file.

        fitted_model_name: str
            Name of the resulting subcircuit, default "s_equivalent"

        create_reference_pins: bool
            If set to True, the synthesized subcircuit will have N pin-pairs:
            p1 p1_ref p2 p2_ref ... pN pN_ref

            If set to False, the synthesized subcircuit will have N pins
            p1 p2 ... pN
            In this case, the reference nodes will be internally connected
            to the global ground net 0.

            The default is False

        Returns
        -------
        None

        Examples
        --------
        Load and fit the `Network`, then export the equivalent subcircuit:

        >>> nw_3port = skrf.Network('my3port.s3p')
        >>> vf = skrf.VectorFitting(nw_3port)
        >>> vf.auto_fit()
        >>> vf.write_spice_subcircuit_s('/my3port_model.sp')

        References
        ----------
        .. [#vf-book] S. Grivet-Talocia and B. Gustavsen, "Passive Macromodeling", Wiley, 2016,
            doi: https://doi.org/10.1002/9781119140931

        """

        if np.any(self.proportional_coeff):
            build_e = True
        else:
            build_e = False

        with open(file, 'w') as f:
            # write title line
            f.write('* EQUIVALENT CIRCUIT FOR VECTOR FITTED S-MATRIX\n')
            f.write('* Created using scikit-rf vectorFitting.py\n')
            f.write('*\n')

            # Create subcircuit pin string and reference nodes
            if create_reference_pins:
                str_input_nodes = " ".join(map(lambda x: f'p{x + 1} p{x + 1}_ref', range(self.network.nports)))
            else:
                str_input_nodes = " ".join(map(lambda x: f'p{x + 1}', range(self.network.nports)))

            f.write(f'.SUBCKT {fitted_model_name} {str_input_nodes}\n')

            for i in range(self.network.nports):
                f.write('*\n')
                f.write(f'* Port network for port {i + 1}\n')

                if create_reference_pins:
                    node_ref_i = f'p{i + 1}_ref'
                else:
                    node_ref_i = '0'

                # reference impedance (real, i.e. resistance) of port i
                z0_i = np.real(self.network.z0[0, i])

                # transfer gain of the controlled current sources representing the incident power wave a_i at port i
                #
                # the gain values result from the definition of the incident power wave:
                # a_i = 1 / 2 / sqrt(Z0_i) * (V_i + Z0_i * I_i) = 1 / 2 / sqrt(Z0_i) * V_i + sqrt(Z0_i) / 2 * I_i
                gain_vccs_a_i = 1 / 2 / np.sqrt(z0_i)
                gain_cccs_a_i = np.sqrt(z0_i) / 2

                # transfer gain of the controlled current source representing the reflected power wave b_i at port i
                #
                # the gain values result from the definition of the reflected power wave:
                # b_i = 1 / 2 / sqrt(Z0_i) * (V_i - Z0_i * I_i)
                #
                # depending on the circuit topology used for the equivalent port network, this can be implemented
                # with either controlled current and/or controlled voltage sources. in case of the Norton current
                # source used in this implementation, the reflected power wave relates to the source current as:
                # b_i = sqrt(Z0_i) / 2 * I_b_i <==> I_b_i = 2 / sqrt(Z0_i) * b_i
                gain_b_i = 2 / np.sqrt(z0_i)

                # dummy voltage source (v = 0) for port current sensing (I_i)
                f.write(f'V{i + 1} p{i + 1} s{i + 1} 0\n')

                # adding port reference resistor Ri = Z0_i
                f.write(f'R{i + 1} s{i + 1} {node_ref_i} {z0_i}\n')

                # transfer of states and inputs from port j to input/output network of port i
                for j in range(self.network.nports):
                    if create_reference_pins:
                        node_ref_j = f'p{j + 1}_ref'
                    else:
                        node_ref_j = '0'

                    # reference impedance (real, i.e. resistance) of port i
                    z0_j = np.real(self.network.z0[0, j])

                    # Stacking order in VectorFitting class variables:
                    # s11, s12, s13, ..., s21, s22, s23, ...
                    idx_S_i_j = i * self.network.nports + j

                    # VCCS and CCCS adding their currents to represent the incident wave a_j
                    gain_vccs_a_j = 1 / 2 / np.sqrt(z0_j)
                    gain_cccs_a_j = np.sqrt(z0_j) / 2

                    d = self.constant_coeff[idx_S_i_j]
                    e = self.proportional_coeff[idx_S_i_j]

                    if d != 0.0:
                        # avoid zero-valued coefficients (in case of fit_constant=False)

                        # input a_j is scaled by constant term d_i_j and by current gain for b_i
                        g_ij = gain_b_i * d * gain_vccs_a_j
                        f_ij = gain_b_i * d * gain_cccs_a_j
                        f.write(f'Gd{i + 1}_{j + 1} {node_ref_i} s{i + 1} p{j + 1} {node_ref_j} {g_ij}\n')
                        f.write(f'Fd{i + 1}_{j + 1} {node_ref_i} s{i + 1} V{j + 1} {f_ij}\n')

                    if build_e and e != 0.0:
                        # avoid zero-valued coefficients (in case of fit_proportional=False)
                        # proportional coefficients require an extra node for the differentiation using an inductor
                        # [Y(s) ~ s * E * U(s)]

                        # differentiated input a_j is scaled by proportional term e_i_j and by current gain for b_i
                        g_ij = gain_b_i * e
                        f.write(f'Ge{i + 1}_{j + 1} {node_ref_i} s{i + 1} e{j + 1} 0 {g_ij}\n')

                    # each residue rk_i_j at port i is multiplied by its respective state signal xk_j
                    for k in range(len(self.poles)):
                        pole = self.poles[k]
                        residue = self.residues[idx_S_i_j, k]
                        g_re = gain_b_i * np.real(residue)
                        g_im = gain_b_i * np.imag(residue)

                        if np.imag(pole) == 0.0:
                            # Real pole/residue pair; represented by one state
                            xkj = f'x{k + 1}_a{j + 1}'
                            f.write(f'Gr{k + 1}_{i + 1}_{j + 1} {node_ref_i} s{i + 1} {xkj} 0 {g_re}\n')
                        else:
                            # Complex-conjugate pole/residue pair; represented by two states
                            # real part at x_{k + 1}_re_{j + 1}
                            # imaginary part at x_{k + 1}_im_{j + 1}
                            xk_re_j = f'x{k + 1}_re_a{j + 1}'
                            xk_im_j = f'x{k + 1}_im_a{j + 1}'
                            f.write(f'Gr{k + 1}_re_{i + 1}_{j + 1} {node_ref_i} s{i + 1} {xk_re_j} 0 {g_re}\n')
                            f.write(f'Gr{k + 1}_im_{i + 1}_{j + 1} {node_ref_i} s{i + 1} {xk_im_j} 0 {g_im}\n')

                # create state networks driven by this port i (input variable u = a_i)
                f.write('*\n')
                f.write(f'* State networks driven by port {i + 1}\n')
                for k in range(len(self.poles)):
                    pole = self.poles[k]
                    pole_re = np.real(pole)
                    pole_im = np.imag(pole)

                    # Transfer of input (a_i) to state networks (node xk_i) using VCCS and CCCS
                    if pole_im == 0.0:
                        # Real pole; represented by one state, input a_i is scaled by b = 1
                        xki = f'x{k + 1}_a{i + 1}'
                        f.write(f'Cx{k + 1}_a{i + 1} {xki} 0 1.0\n')  # 1F capacitor makes math easy
                        f.write(f'Gx{k + 1}_a{i + 1} 0 {xki} p{i + 1} {node_ref_i} {1 * gain_vccs_a_i}\n')
                        f.write(f'Fx{k + 1}_a{i + 1} 0 {xki} V{i + 1} {1 * gain_cccs_a_i}\n')
                        f.write(f'Rp{k + 1}_a{i + 1} 0 {xki} {-1 / pole_re}\n')
                    else:
                        # Complex pole of a conjugate pair; represented by two states
                        # real part at x_{k + 1}_re_{i + 1}, input a_i is scaled by b = 2
                        xk_re_i = f'x{k + 1}_re_a{i + 1}'
                        xk_im_i = f'x{k + 1}_im_a{i + 1}'
                        f.write(f'Cx{k + 1}_re_a{i + 1} {xk_re_i} 0 1.0\n')  # 1F capacitor makes math easy
                        f.write(
                            f'Gx{k + 1}_re_a{i + 1} 0 {xk_re_i} p{i + 1} {node_ref_i} {2 * gain_vccs_a_i}\n')
                        f.write(f'Fx{k + 1}_re_a{i + 1} 0 {xk_re_i} V{i + 1} {2 * gain_cccs_a_i}\n')
                        f.write(f'Rp{k + 1}_re_re_a{i + 1} 0 {xk_re_i} {-1 / pole_re}\n')
                        f.write(f'Gp{k + 1}_re_im_a{i + 1} 0 {xk_re_i} {xk_im_i} 0 {pole_im}\n')

                        # imaginary part at x_{k + 1}_im_{i + 1}, input a_i is inactive (b = 0)
                        f.write(f'Cx{k + 1}_im_a{i + 1} {xk_im_i} 0 1.0\n')  # 1F capacitor makes math easy
                        f.write(f'Gp{k + 1}_im_re_a{i + 1} 0 {xk_im_i} {xk_re_i} 0 {-1 * pole_im}\n')
                        f.write(f'Rp{k + 1}_im_im_a{i + 1} 0 {xk_im_i} {-1 / pole_re}\n')

                # create differentiation network for this port i (input variable u = a_i)
                if build_e:
                    f.write('*\n')
                    f.write(f'* Network with derivative of input a_{i + 1} for proportional term\n')
                    # voltage on node 'e{i + 1}' to gnd (0) represents time-derivative of input a_i for terms e_j_i
                    f.write(f'Le{i + 1} e{i + 1} 0 1.0\n')  # 1H inductor makes math easy
                    f.write(f'Ge{i + 1} 0 e{i + 1} p{i + 1} {node_ref_i} {gain_vccs_a_i}\n')
                    f.write(f'Fe{i + 1} 0 e{i + 1} V{i + 1} {gain_cccs_a_i}\n')

            f.write(f'.ENDS {fitted_model_name}\n')
