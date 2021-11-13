import numpy as np
import os

# imports for type hinting
from typing import Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from .network import Network

from functools import wraps
try:
    from . import plotting    # will perform the correct setup for matplotlib before it is called below
    import matplotlib.pyplot as mplt
    from matplotlib.ticker import EngFormatter
except ImportError:
    mplt = None

import logging
import warnings
from timeit import default_timer as timer


def check_plotting(func):
    """
    This decorator checks if matplotlib.pyplot is available under the name mplt.
    If not, raise an RuntimeError.

    Raises
    ------
    RuntimeError
        When trying to run the decorated function without matplotlib
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if mplt is None:
            raise RuntimeError('Plotting is not available')
        func(*args, **kwargs)

    return wrapper


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

    def __init__(self, network: 'Network'):
        self.network = network
        self.initial_poles = None

        self.poles = None
        """ Instance variable holding the list of fitted poles. Will be initialized by :func:`vector_fit`. """

        self.zeros = None
        """ Instance variable holding the list of fitted zeros. Will be initialized by :func:`vector_fit`. """

        self.proportional_coeff = None
        """ Instance variable holding the list of fitted proportional coefficients. Will be initialized by 
        :func:`vector_fit`. """

        self.constant_coeff = None
        """ Instance variable holding the list of fitted constants. Will be initialized by :func:`vector_fit`. """

        self.max_iterations = 100
        """ Instance variable specifying the maximum number of iterations for the fitting process. To be changed by the
         user before calling :func:`vector_fit`. """

        self.max_tol = 1e-6
        """ Instance variable specifying the convergence criterion in terms of relative tolerance. To be changed by the
         user before calling :func:`vector_fit`. """

        self.wall_clock_time = 0
        """ Instance variable holding the wall-clock time (in seconds) consumed by the most recent fitting process in 
        :func:`vector_fit`. Subsequent calls of :func:`vector_fit` will overwrite this value. """

        self.d_res_history = []
        self.delta_max_history = []
        self.history_max_sigma = []
        self.history_cond_A = []

    def vector_fit(self, n_poles_real: int = 2, n_poles_cmplx: int = 2, init_pole_spacing: str = 'lin',
                   parameter_type: str = 's', fit_constant: bool = True, fit_proportional: bool = False) -> None:
        """
        Main work routine performing the vector fit. The results will be stored in the class variables
        :attr:`poles`, :attr:`zeros`, :attr:`proportional_coeff` and :attr:`constant_coeff`.

        Parameters
        ----------
        n_poles_real : int, optional
            Number of initial real poles. See notes.

        n_poles_cmplx : int, optional
            Number of initial complex conjugate poles. See notes.

        init_pole_spacing : str, optional
            Type of initial pole spacing across the frequency interval of the S-matrix. Either linear (lin) or
            logarithmic (log).

        parameter_type : str, optional
            Representation type of the frequency responses to be fitted. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`). As scikit-rf can currently
            only read S parameters from a Touchstone file, the fit should also be performed on the original S
            parameters. Otherwise, scikit-rf will convert the responses from S to Z or Y, which might work for the fit
            but can cause other issues.

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
        """

        timer_start = timer()

        # create initial poles and space them across the frequencies in the provided Touchstone file
        # use normalized frequencies during the iterations (seems to be more stable during least-squares fit)
        norm = np.average(self.network.f)
        freqs_norm = np.array(self.network.f) / norm

        fmin = np.amin(freqs_norm)
        fmax = np.amax(freqs_norm)
        if init_pole_spacing == 'log':
            pole_freqs_real = np.geomspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.geomspace(fmin, fmax, n_poles_cmplx)
        elif init_pole_spacing == 'lin':
            pole_freqs_real = np.linspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.linspace(fmin, fmax, n_poles_cmplx)
        else:
            logging.warning('Invalid choice of initial pole spacing; proceeding with linear spacing')
            pole_freqs_real = np.linspace(fmin, fmax, n_poles_real)
            pole_freqs_cmplx = np.linspace(fmin, fmax, n_poles_cmplx)

        # init poles array of correct length
        poles_re = np.zeros(n_poles_real + n_poles_cmplx)
        poles_im = np.zeros(n_poles_real + n_poles_cmplx)

        # add real poles
        for i, f in enumerate(pole_freqs_real):
            omega = 2 * np.pi * f
            poles_re[i] = -1 * omega

        # add complex-conjugate poles (store only positive imaginary parts)
        i_offset = len(pole_freqs_real)
        for i, f in enumerate(pole_freqs_cmplx):
            omega = 2 * np.pi * f
            poles_re[i_offset + i] = -1 / 100 * omega
            poles_im[i_offset + i] = omega

        # save initial poles (un-normalize first)
        self.initial_poles = (poles_re + 1j * poles_im) * norm
        max_singular = 1

        logging.info('### Starting pole relocation process.\n')

        # stack frequency responses as a single vector
        # stacking order (row-major):
        # s11, s12, s13, ..., s21, s22, s23, ...
        freq_responses = []
        for i in range(self.network.nports):
            for j in range(self.network.nports):
                if parameter_type.lower() == 's':
                    freq_responses.append(self.network.s[:, i, j])
                elif parameter_type.lower() == 'z':
                    freq_responses.append(self.network.z[:, i, j])
                elif parameter_type.lower() == 'y':
                    freq_responses.append(self.network.y[:, i, j])
                else:
                    logging.warning('Invalid choice of matrix parameter type (S, Z, or Y); proceeding with S representation.')
                    freq_responses.append(self.network.s[:, i, j])
        freq_responses = np.array(freq_responses)

        # ITERATIVE FITTING OF POLES to the provided frequency responses
        # initial set of poles will be replaced with new poles after every iteration
        iterations = self.max_iterations
        self.d_res_history = []
        self.delta_max_history = []
        self.history_cond_A = []
        converged = False
        while iterations > 0:
            logging.info('Iteration {}'.format(self.max_iterations - iterations + 1))

            # count number of rows and columns in final coefficient matrix to solve for (c_res, d_res)
            # (ratio #real/#complex poles might change during iterations)
            n_cols_unused = 0
            for pole in poles_im:
                if pole == 0.0:
                    n_cols_unused += 1
                else:
                    n_cols_unused += 2
            n_cols_used = n_cols_unused
            n_cols_used += 1
            if fit_constant:
                n_cols_unused += 1
            if fit_proportional:
                n_cols_unused += 1
            n_rows_A = n_cols_used * len(freq_responses)

            # generate coefficients of approximation function for each target frequency response
            # responses will be reduced independently using QR decomposition
            # simplified coeff. matrices of all responses will be stacked in matrix A for least-squares solver
            A = np.empty((n_rows_A, n_cols_used))
            b = np.zeros(n_rows_A)

            for i_response, freq_response in enumerate(freq_responses):
                # calculate coefficients for each frequency response
                # A_sub which will be reduced first (QR decomposition) and then filled into the main
                # coefficient matrix A
                # each frequency sample s_k of the target response generates two rows in the matrix
                # (1st for real part, 2nd for imaginary part)

                A_sub = np.empty((2 * len(freqs_norm), n_cols_unused + n_cols_used))
                A_row_extra = np.zeros(n_cols_used)

                # responses will be weighted according to their norm;
                # alternative: equal weights with weight_response = 1.0
                # or anti-proportional weights with weight_response = 1 / np.linalg.norm(freq_response)
                weight_response = np.linalg.norm(freq_response)

                for k, f_sample in enumerate(freqs_norm):
                    # this loop is performance critical, as it gets repeated many times (N_responses * N_frequencies)
                    omega_k = 2 * np.pi * f_sample
                    resp_re = np.real(freq_response[k])
                    resp_im = np.imag(freq_response[k])
                    i_col = 0
                    i_row_re = 2 * k
                    i_row_im = i_row_re + 1

                    # layout of this row in A_sub:
                    # [pole1, pole2, ..., (constant), (proportional), pole1, pole2, ..., constant]
                    # to avoid looping through the poles twice, both loops are merged (faster, but less comprehensible)

                    # add coefficients for a pair of complex conjugate poles
                    # part 1: first sum of rational functions (residue variable c)
                    # merged with
                    # part 3: second sum of rational functions (variable c_res)
                    for i_pole in range(len(poles_im)):
                        pole_re = poles_re[i_pole]
                        pole_im = poles_im[i_pole]

                        if pole_im == 0.0:
                            # coefficient for a real pole: p = p'
                            denom = (omega_k ** 2 + pole_re ** 2)
                            coeff_re = -1 * pole_re / denom
                            coeff_im = -1 * omega_k / denom

                            # part 1: coeff = 1 / (s_k - p') = coeff_re + j coeff_im
                            A_sub[i_row_re, i_col] = coeff_re
                            A_sub[i_row_im, i_col] = coeff_im
                            # part 3: coeff = -1 * H(s_k) / (s_k - pole)
                            # Re{coeff} = -1 * coeff_re * resp_re + coeff_im * resp_im
                            # Im{coeff} = -1 * coeff_re * resp_im - coeff_im * resp_re
                            A_sub[i_row_re, n_cols_unused + i_col] = -1 * coeff_re * resp_re + coeff_im * resp_im
                            A_sub[i_row_im, n_cols_unused + i_col] = -1 * coeff_re * resp_im - coeff_im * resp_re
                            # extra equation to avoid trivial solution:
                            # coeff += Re(1 / (s_k - pole)) = coeff_re
                            A_row_extra[i_col] += coeff_re
                            i_col += 1
                        else:
                            # coefficient for a complex pole of a conjugated pair: p = p' + jp''
                            denom1 = pole_re ** 2 + (omega_k - pole_im) ** 2
                            denom2 = pole_re ** 2 + (omega_k + pole_im) ** 2

                            # row 1: add coefficient for real part of residue
                            # part 1: coeff = 1 / (s_k - pole) + 1 / (s_k - conj(pole))
                            coeff_re = -1 * pole_re * (1 / denom1 + 1 / denom2)
                            coeff_im = (pole_im - omega_k) / denom1 - (pole_im + omega_k) / denom2
                            A_sub[i_row_re, i_col] = coeff_re
                            A_sub[i_row_im, i_col] = coeff_im
                            # extra equation to avoid trivial solution:
                            # coeff += Re{1 / (s_k - pole) + 1 / (s_k - conj(pole))}
                            A_row_extra[i_col] += coeff_re
                            # part 3: coeff = -1 * H(s_k) * [1 / (s_k - pole) + 1 / (s_k - conj(pole))]
                            A_sub[i_row_re, n_cols_unused + i_col] = -1 * coeff_re * resp_re + coeff_im * resp_im
                            A_sub[i_row_im, n_cols_unused + i_col] = -1 * coeff_re * resp_im - coeff_im * resp_re
                            i_col += 1

                            # row 2: add coefficient for imaginary part of residue
                            # part 1: coeff = 1j / (s_k - pole) - 1j / (s_k - conj(pole))
                            #               = (p'' - omega + j p') * (1 / denom2 - 1 / denom1)
                            coeff_re = (omega_k - pole_im) / denom1 - (omega_k + pole_im) / denom2
                            coeff_im = pole_re * (1 / denom2 - 1 / denom1)
                            A_sub[i_row_re, i_col] = coeff_re
                            A_sub[i_row_im, i_col] = coeff_im
                            # extra equation to avoid trivial solution:
                            # coeff += Re(1j / (s_k - pole) - 1j / (s_k - conj(pole)))
                            A_row_extra[i_col] += coeff_re
                            # part 3: coeff = -1 * H(s_k) * [1j / (s_k - pole) - 1j / (s_k - conj(pole))]
                            A_sub[i_row_re, n_cols_unused + i_col] = -1 * coeff_re * resp_re + coeff_im * resp_im
                            A_sub[i_row_im, n_cols_unused + i_col] = -1 * coeff_re * resp_im - coeff_im * resp_re
                            i_col += 1

                    # part 4: constant (variable d_res)
                    # coeff = -1 * H(s_k)
                    A_sub[i_row_re, n_cols_unused + i_col] = -1 * resp_re
                    A_sub[i_row_im, n_cols_unused + i_col] = -1 * resp_im
                    # extra equation to avoid trivial solution: coeff += 1
                    A_row_extra[i_col] += 1

                    # part 2: constant (variable d) and proportional term (variable e)
                    if fit_constant:
                        # coeff = 1 + j0
                        A_sub[i_row_re, i_col] = 1.0
                        A_sub[i_row_im, i_col] = 0.0
                        i_col += 1
                    if fit_proportional:
                        # coeff = s_k = j omega_k
                        A_sub[i_row_re, i_col] = 0.0
                        A_sub[i_row_im, i_col] = omega_k

                # QR decomposition
                Q, R = np.linalg.qr(A_sub, 'reduced')

                # only R22 is required to solve for c_res and d_res
                R22 = R[n_cols_unused:, n_cols_unused:]

                # similarly, only right half of Q is required (not used here, because RHS is zero)
                # Q2 = Q[:, n_cols_unused:]

                # apply weight of this response and add coefficients to the system matrix
                A[i_response * n_cols_used:(i_response + 1) * n_cols_used, :] = np.sqrt(weight_response) * R22
                # multiplication of Q2 by rhs=0 omitted; right-hand side would also require weighting
                # b[i_response * n_cols_used:(i_response + 1) * n_cols_used] = np.matmul(np.transpose(Q2), rhs)

                # add extra equation to avoid trivial solution
                weight_extra = np.linalg.norm(weight_response * freq_response) / len(freq_response)
                A[(i_response + 1) * n_cols_used - 1, :] = np.sqrt(weight_extra) * A_row_extra
                b[(i_response + 1) * n_cols_used - 1] = np.sqrt(weight_extra) * len(freq_response)

            cond_A = np.linalg.cond(A)
            logging.info('Condition number of coeff. matrix A = {}'.format(cond_A))
            self.history_cond_A.append(cond_A)

            # solve least squares for real parts
            x, residuals, rank, singular_vals = np.linalg.lstsq(A, b, rcond=None)

            # assemble individual result vectors from single LS result x
            c_res = x[:-1]
            d_res = x[-1]

            # check if d_res is suited for zeros calculation
            tol_res = 1e-8
            if np.abs(d_res) < tol_res:
                # d_res is too small, discard solution and proceed the |d_res| = tol_res
                d_res = tol_res * (d_res / np.abs(d_res))
                warnings.warn('Replacing d_res solution as it was too small. This is not a good sign and probably '
                              'means that more starting poles are required', RuntimeWarning)

            self.d_res_history.append(d_res)
            logging.info('d_res = {}'.format(d_res))

            # build test matrix H, which will hold the new poles as eigenvalues
            H = np.zeros((len(c_res), len(c_res)))
            i = 0
            for i_pole in range(len(poles_im)):
                # fill diagonal with previous poles
                pole_re = poles_re[i_pole]
                pole_im = poles_im[i_pole]
                if pole_im == 0.0:
                    # one row for a real pole
                    H[i, i] = pole_re
                    H[i] -= c_res / d_res
                    i += 1
                else:
                    # two rows for a complex pole of a conjugated pair
                    H[i, i] = pole_re
                    H[i, i + 1] = pole_im
                    H[i + 1, i] = -1 * pole_im
                    H[i + 1, i + 1] = pole_re
                    H[i] -= 2 * c_res / d_res
                    i += 2
            poles_new = np.linalg.eigvals(H)

            # replace poles for next iteration
            poles_re = []
            poles_im = []
            for k, pole in enumerate(poles_new):
                pole_re = np.real(pole)
                pole_im = np.imag(pole)
                if pole_re > 0.0:
                    # flip real part of unstable poles (real part needs to be negative for stability)
                    pole_re = -1 * pole_re
                if pole_im >= 0.0:
                    # complex poles need to come in complex conjugate pairs; append only the positive part
                    poles_re.append(pole_re)
                    poles_im.append(pole_im)

            # calculate relative changes in the singular values; stop iteration loop once poles have converged
            new_max_singular = np.amax(singular_vals)
            delta_max = np.abs(1 - new_max_singular / max_singular)
            self.delta_max_history.append(delta_max)
            logging.info('Max. relative change in residues = {}\n'.format(delta_max))
            max_singular = new_max_singular

            stop = False
            if delta_max < self.max_tol:
                if converged:
                    # is really converged, finish
                    logging.info('Pole relocation process converged after {} iterations.'.format(
                        self.max_iterations - iterations + 1))
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
                max_cond = np.amax(self.history_cond_A)
                if max_cond > 1e10:
                    msg_illcond = 'Hint: the linear system was ill-conditioned (max. condition number = {}). ' \
                                  'This often means that more poles are required.'.format(max_cond)
                else:
                    msg_illcond = ''
                if converged and stop is False:
                    warnings.warn('Vector Fitting: The pole relocation process barely converged to tolerance. '
                                  'It took the max. number of iterations (N_max = {}). '
                                  'The results might not have converged properly. '.format(self.max_iterations)
                                  + msg_illcond, RuntimeWarning)
                else:
                    warnings.warn('Vector Fitting: The pole relocation process stopped after reaching the '
                                  'maximum number of iterations (N_max = {}). '
                                  'The results did not converge properly. '.format(self.max_iterations)
                                  + msg_illcond, RuntimeWarning)

            if stop:
                iterations = 0

        # ITERATIONS DONE
        poles = np.array(poles_re) + 1j * np.array(poles_im)

        logging.info('Initial poles before relocation:')
        logging.info(self.initial_poles)

        logging.info('Final poles:')
        logging.info(poles * norm)

        logging.info('\n### Starting zeros calculation process.\n')

        # finally, solve for the residues with the previously calculated poles
        zeros = []
        constant_coeff = []
        proportional_coeff = []

        for freq_response in freq_responses:
            # calculate coefficients (row A_k in matrix) for each frequency sample s_k of the target response
            # row will be appended to submatrix A_sub of complete coeff matrix A_matrix
            # 2 rows per pole in result vector (1st for real part, 2nd for imaginary part)
            # --> 2 columns per pole in coeff matrix
            n_cols = 0
            for pole_im in poles_im:
                if pole_im == 0.0:
                    n_cols += 1
                else:
                    n_cols += 2
            if fit_constant:
                n_cols += 1
            if fit_proportional:
                n_cols += 1
            A_matrix = np.empty((2 * len(freqs_norm), n_cols))
            b_vector = np.empty((2 * len(freqs_norm)))

            for k, f_sample in enumerate(freqs_norm):
                omega_k = 2 * np.pi * f_sample
                resp_re = np.real(freq_response[k])
                resp_im = np.imag(freq_response[k])
                i_row_re = 2 * k
                i_row_im = i_row_re + 1
                i_col = 0

                # add coefficients for a pair of complex conjugate poles
                # part 1: first sum of rational functions (residue variable c)
                for i_pole in range(len(poles_im)):
                    pole_re = poles_re[i_pole]
                    pole_im = poles_im[i_pole]

                    # separate and stack real and imaginary part to preserve conjugacy of the pole pair
                    if pole_im == 0.0:
                        denom = pole_re ** 2 + omega_k ** 2
                        A_matrix[i_row_re, i_col] = -1 * pole_re / denom
                        A_matrix[i_row_im, i_col] = -1 * omega_k / denom
                        i_col += 1
                    else:
                        # coefficient for real part of residue
                        denom1 = pole_re ** 2 + (omega_k - pole_im) ** 2
                        denom2 = pole_re ** 2 + (omega_k + pole_im) ** 2
                        A_matrix[i_row_re, i_col] = -1 * pole_re * (1 / denom1 + 1 / denom2)
                        A_matrix[i_row_im, i_col] = (pole_im - omega_k) / denom1 - (pole_im + omega_k) / denom2
                        i_col += 1
                        # coefficient for imaginary part of residue
                        A_matrix[i_row_re, i_col] = (omega_k - pole_im) / denom1 - (pole_im + omega_k) / denom2
                        A_matrix[i_row_im, i_col] = pole_re * (1 / denom2 - 1 / denom1)
                        i_col += 1

                # part 2: constant (variable d) and proportional term (variable e)
                if fit_constant:
                    A_matrix[i_row_re, i_col] = 1.0
                    A_matrix[i_row_im, i_col] = 0.0
                    i_col += 1
                if fit_proportional:
                    A_matrix[i_row_re, i_col] = 0.0
                    A_matrix[i_row_im, i_col] = omega_k

                b_vector[i_row_re] = resp_re
                b_vector[i_row_im] = resp_im

            logging.info('A_matrix: condition number = {}'.format(np.linalg.cond(A_matrix)))

            # solve least squares and obtain results as stack of real part vector and imaginary part vector
            x, residuals, rank, singular_vals = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

            i = 0
            zeros_response = []
            for i_pole in range(len(poles_im)):
                pole_im = poles_im[i_pole]

                if pole_im == 0.0:
                    zeros_response.append(x[i] + 0j)
                    i += 1
                else:
                    zeros_response.append(x[i] + 1j * x[i + 1])
                    i += 2
            zeros.append(zeros_response)

            if fit_constant and fit_proportional:
                # both constant d and proportional e were fitted
                constant_coeff.append(x[-2])
                proportional_coeff.append(x[-1])
            elif fit_constant:
                # only constant d was fitted
                constant_coeff.append(x[-1])
                proportional_coeff.append(0.0)
            elif fit_proportional:
                # only proportional e was fitted
                constant_coeff.append(0.0)
                proportional_coeff.append(x[-1])
            else:
                # neither constant d nor proportional e was fitted
                constant_coeff.append(0.0)
                proportional_coeff.append(0.0)

        # save poles, zeros, d, e in actual frequencies (un-normalized)
        self.poles = poles * norm
        self.zeros = np.array(zeros) * norm
        self.constant_coeff = np.array(constant_coeff)
        self.proportional_coeff = np.array(proportional_coeff) / norm

        timer_stop = timer()
        self.wall_clock_time = timer_stop - timer_start

        logging.info('\n### Vector fitting finished in {} seconds.\n'.format(self.wall_clock_time))

    def get_rms_error(self, i=-1, j=-1, parameter_type: str = 's'):
        """
        Returns the root-mean-square (rms) error magnitude of the fit, i.e.
        :math:`\sqrt{ \mathrm{mean}(|S_{i,j} - S_{i,j,\mathrm{fit}} |^2) }`,
        either for an individual response :math:`S_{i,j}` or for larger slices of the network.

        Parameters
        ----------
        i : int, optional
            Row indices of the responses to be evaluated. Either a single row selected by an integer
            :math:`i \in [0, N_\mathrm{ports}]`, or multiple rows selected by a list of integers, or all rows
            selected by :math:`i = -1` (*default*).

        j : int, optional
            Column indices of the responses to be evaluated. Either a single column selected by an integer
            :math:`j \in [0, N_\mathrm{ports}]`, or multiple columns selected by a list of integers, or all columns
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
            raise ValueError('Invalid parameter type \'{}\'. Valid options: \`s\`, \`z\`, or \`y\`'.format(parameter_type))

        error_mean_squared = 0
        for i in list_i:
            for j in list_j:
                nw_ij = nw_responses[:, i, j]
                fit_ij = self.get_model_response(i, j, self.network.f)
                error_mean_squared += np.mean(np.square(np.abs(nw_ij - fit_ij)))

        return np.sqrt(error_mean_squared)

    def _get_ABCDE(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            State-space matrix C holding the residues (zeros)
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
        if self.zeros is None:
            raise ValueError('self.zeros = None; nothing to do. You need to run vector_fit() first.')
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

        # state-space matrix C holds the residues (zeros)
        # assemble C = [[R1.11, R1.12, R1.13, ...], [R2.11, R2.12, R2.13, ...], ...]
        C = np.zeros(shape=(n_ports, n_matrix))
        for i in range(n_ports):
            for j in range(n_ports):
                # i: row index
                # j: column index
                i_response = i * n_ports + j

                j_zeros = 0
                for zero in self.zeros[i_response]:
                    if np.imag(zero) == 0.0:
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_zeros] = np.real(zero)
                        j_zeros += 1
                    else:
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_zeros] = np.real(zero)
                        C[i, j * (n_poles_real + 2 * n_poles_cplx) + j_zeros + 1] = np.imag(zero)
                        j_zeros += 2

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
    def _get_s_from_ABCDE(freq: float,
                          A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Private method.
        Returns the S-matrix of the vector fitted model calculated from the real-valued system matrices of the state-
        space representation, as provided by `_get_ABCDE()`.

        Parameters
        ----------
        freq : float
            Frequency (in Hz) at which to calculate the S-matrix.
        A : ndarray
        B : ndarray
        C : ndarray
        D : ndarray
        E : ndarray

        Returns
        -------
        ndarray
            Complex-valued S-matrix (NxN) calculated at frequency `freq`.
        """

        dim_A = np.shape(A)[0]
        stsp_poles = np.linalg.inv(2j * np.pi * freq * np.identity(dim_A) - A)
        stsp_S = np.matmul(np.matmul(C, stsp_poles), B)
        stsp_S += D + 2j * np.pi * freq * E
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
                             'not make any sense; you need to run vector_fit() with option \'fit_proportional=False\' '
                             'first.')

        # # the network needs to be reciprocal for this passivity test method to work: S = transpose(S)
        # if not np.allclose(self.zeros, np.transpose(self.zeros)) or \
        #         not np.allclose(self.constant_coeff, np.transpose(self.constant_coeff)) or \
        #         not np.allclose(self.proportional_coeff, np.transpose(self.proportional_coeff)):
        #     logging.error('Passivity testing with unsymmetrical model parameters is not supported. '
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

        # sort the output from lower to higher frequencies
        freqs_violation = np.sort(freqs_violation)

        # identify frequency bands of passivity violations

        # sweep the bands between crossover frequencies and identify bands of passivity violations
        violation_bands = []
        for i, freq in enumerate(freqs_violation):
            if i == 0:
                f_start = 0
                f_stop = freq
            else:
                f_start = freqs_violation[i - 1]
                f_stop = freq

            # calculate singular values at the center frequency between crossover frequencies to identify violations
            f_center = 0.5 * (f_start + f_stop)
            s_center = self._get_s_from_ABCDE(f_center, A, B, C, D, E)
            u, sigma, vh = np.linalg.svd(s_center)
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
        """

        viol_bands = self.passivity_test(parameter_type)
        if len(viol_bands) == 0:
            return True
        else:
            return False

    def passivity_enforce(self, n_samples: int = 100, parameter_type: str = 's') -> None:
        """
        Enforces the passivity of the vector fitted model, if required. This is an implementation of the method
        presented in [#]_.

        Parameters
        ----------
        n_samples : int, optional
            Number of linearly spaced frequency samples at which passivity will be evaluated and enforce. (Default: 100)

        parameter_type : str, optional
            Representation type of the fitted frequency responses. Either *scattering* (:attr:`s` or :attr:`S`),
            *impedance* (:attr:`z` or :attr:`Z`) or *admittance* (:attr:`y` or :attr:`Y`). Currently, only scattering
            parameters are supported for passivity evaluation.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the function is called for `parameter_type` different than `S` (scattering).

        ValueError
            If the function is used with a model containing nonzero proportional coefficients.

        See Also
        --------
        is_passive : Returns the passivity status of the model as a boolean value.
        passivity_test : Verbose passivity evaluation routine.
        plot_passivation : Convergence plot for passivity enforcement iterations.

        References
        ----------
        .. [#] T. Dhaene, D. Deschrijver and N. Stevens, "Efficient Algorithm for Passivity Enforcement of S-Parameter-
            Based Macromodels," in IEEE Transactions on Microwave Theory and Techniques, vol. 57, no. 2, pp. 415-420,
            Feb. 2009, DOI: 10.1109/TMTT.2008.2011201.
        """

        if parameter_type.lower() != 's':
            raise NotImplementedError('Passivity testing is currently only supported for scattering (S) parameters.')
        if parameter_type.lower() == 's' and len(np.flatnonzero(self.proportional_coeff)) > 0:
            raise ValueError('Passivity testing of scattering parameters with nonzero proportional coefficients does '
                             'not make any sense; you need to run vector_fit() with option \'fit_proportional=False\' '
                             'first.')

        # always run passivity test first; this will write 'self.violation_bands'
        violation_bands = self.passivity_test()

        if len(violation_bands) == 0:
            # model is already passive; do nothing and return
            logging.info('Passivity enforcement: The model is already passive. Nothing to do.')
            return

        freqs_eval = np.linspace(0, 1.2 * violation_bands[-1, -1], n_samples)
        A, B, C, D, E = self._get_ABCDE()
        dim_A = np.shape(A)[0]
        C_t = C
        delta = 0.999   # predefined tolerance parameter (users should not need to change this)

        # iterative compensation of passivity violations
        t = 0
        self.history_max_sigma = []
        while t < self.max_iterations:
            logging.info('Passivity enforcement; Iteration {}'.format(t + 1))

            A_matrix = []
            b_vector = []
            sigma_max = 0

            # sweep through evaluation frequencies
            for i_eval, freq_eval in enumerate(freqs_eval):
                # calculate S-matrix at this frequency
                s_eval = self._get_s_from_ABCDE(freq_eval, A, B, C_t, D, E)

                # singular value decomposition
                u, sigma, vh = np.linalg.svd(s_eval)

                # keep track of the greatest singular value in every iteration step
                if np.amax(sigma) > sigma_max:
                    sigma_max = np.amax(sigma)

                # prepare and fill the square matrices 'gamma' and 'psi' marking passivity violations
                gamma = np.diag(sigma)
                psi = np.diag(sigma)
                for i, sigma_i in enumerate(sigma):
                    if sigma_i <= delta:
                        gamma[i, i] = 0
                        psi[i, i] = 0
                    else:
                        gamma[i, i] = 1
                        psi[i, i] = delta

                # calculate violation S-matrix
                # s_viol is again a complex NxN S-matrix (N: number of network ports)
                sigma_viol = np.matmul(np.diag(sigma), gamma) - psi
                s_viol = np.matmul(np.matmul(u, sigma_viol), vh)

                # Laplace frequency of this sample in the sweep
                s_k = 2j * np.pi * freq_eval

                # build matrix system for least-squares fitting of new set of violation residues C_viol
                # using rule for transpose of matrix products: transpose(A * B) = transpose(B) * transpose(A)
                # hence, S = C * coeffs <===> transpose(S) = transpose(coeffs) * transpose(C)
                coeffs = np.transpose(np.matmul(np.linalg.inv(s_k * np.identity(dim_A) - A), B))
                if i_eval == 0:
                    A_matrix = np.vstack([np.real(coeffs), np.imag(coeffs)])
                    b_vector = np.vstack([np.real(np.transpose(s_viol)), np.imag(np.transpose(s_viol))])
                else:
                    A_matrix = np.concatenate((A_matrix, np.vstack([np.real(coeffs),
                                                                    np.imag(coeffs)])), axis=0)
                    b_vector = np.concatenate((b_vector, np.vstack([np.real(np.transpose(s_viol)),
                                                                    np.imag(np.transpose(s_viol))])), axis=0)

            # solve least squares
            x, residuals, rank, singular_vals = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

            C_viol = np.transpose(x)

            # calculate and update C_t for next iteration
            C_t = C_t - C_viol
            t += 1
            self.history_max_sigma.append(sigma_max)

            # stop iterations when model is passive
            if sigma_max < 1.0:
                break

        # PASSIVATION PROCESS DONE; model is either passive or max. number of iterations have been exceeded
        if t == self.max_iterations:
            logging.error('Passivity enforcement: Aborting after the max. number of iterations has been exceeded.')

        # save/update model parameters (perturbed residues)
        self.history_max_sigma = np.array(self.history_max_sigma)

        n_ports = np.shape(D)[0]
        for i in range(n_ports):
            k = 0   # column index in C_t
            for j in range(n_ports):
                i_response = i * n_ports + j
                z = 0   # column index self.zeros
                for pole in self.poles:
                    if np.imag(pole) == 0.0:
                        # real pole --> real residue
                        self.zeros[i_response, z] = C_t[i, k]
                        k += 1
                    else:
                        # complex-conjugate pole --> complex-conjugate residue
                        self.zeros[i_response, z] = C_t[i, k] + 1j * C_t[i, k + 1]
                        k += 2
                    z += 1

    def write_npz(self, path: str) -> None:
        """
        Writes the model parameters in :attr:`poles`, :attr:`zeros`,
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
        """

        if self.poles is None:
            logging.error('Nothing to export; Poles have not been fitted.')
            return
        if self.zeros is None:
            logging.error('Nothing to export; Zeros have not been fitted.')
            return
        if self.proportional_coeff is None:
            logging.error('Nothing to export; Proportional coefficients have not been fitted.')
            return
        if self.constant_coeff is None:
            logging.error('Nothing to export; Constants have not been fitted.')
            return

        filename = self.network.name

        logging.warning('Exporting results as compressed NumPy array to {}'.format(path))
        np.savez_compressed(os.path.join(path, 'coefficients_{}'.format(filename)),
                            poles=self.poles, zeros=self.zeros, proportionals=self.proportional_coeff,
                            constants=self.constant_coeff)

    def read_npz(self, file: str) -> None:
        """
        Reads all model parameters :attr:`poles`, :attr:`zeros`, :attr:`proportional_coeff` and :attr:`constant_coeff`
        from a labeled NumPy .npz file.

        Parameters
        ----------
        file : str
            NumPy .npz file containing the parameters. See notes.

        Returns
        -------
        None

        Notes
        -----
        The .npz file needs to include the model parameters as individual NumPy arrays (ndarray) labeled '*poles*',
        '*zeros*', '*proportionals*' and '*constants*'. The shapes of those arrays need to match the network properties
        in :class:`network` (correct number of ports). Preferably, the .npz file was created by :func:`write_npz`.

        See Also
        --------
        write_npz : Writes all model parameters to a .npz file
        """

        with np.load(file) as data:
            poles = data['poles']
            zeros = data['zeros']
            proportional_coeff = data['proportionals']
            constant_coeff = data['constants']

            n_ports = int(np.sqrt(len(constant_coeff)))
            n_resp = n_ports ** 2
            if np.shape(zeros)[0] == np.shape(proportional_coeff)[0] == np.shape(constant_coeff)[0] == n_resp:
                self.poles = poles
                self.zeros = zeros
                self.proportional_coeff = proportional_coeff
                self.constant_coeff = constant_coeff
            else:
                logging.error('Length of the provided parameters does not match the network size.')

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
        """

        if self.poles is None:
            logging.error('Returning zero; Poles have not been fitted.')
            return np.zeros_like(freqs)
        if self.zeros is None:
            logging.error('Returning zero; Zeros have not been fitted.')
            return np.zeros_like(freqs)
        if self.proportional_coeff is None:
            logging.error('Returning zero; Proportional coefficients have not been fitted.')
            return np.zeros_like(freqs)
        if self.constant_coeff is None:
            logging.error('Returning zero; Constants have not been fitted.')
            return np.zeros_like(freqs)
        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        s = 2j * np.pi * np.array(freqs)
        n_ports = int(np.sqrt(len(self.constant_coeff)))
        i_response = i * n_ports + j
        zeros = self.zeros[i_response]

        resp = self.proportional_coeff[i_response] * s + self.constant_coeff[i_response]
        for i, pole in enumerate(self.poles):
            if np.imag(pole) == 0.0:
                # real pole
                resp += zeros[i] / (s - pole)
            else:
                # complex conjugate pole
                resp += zeros[i] / (s - pole) + np.conjugate(zeros[i]) / (s - np.conjugate(pole))
        return resp

    @check_plotting
    def plot_s_db(self, i: int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the magnitude in dB of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, 20 * np.log10(np.abs(self.network.s[:, i, j])), color='r', label='Samples')
        ax.plot(freqs, 20 * np.log10(np.abs(self.get_model_response(i, j, freqs))), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_mag(self, i: int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the magnitude in linear scale of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, np.abs(self.network.s[:, i, j]), color='r', label='Samples')
        ax.plot(freqs, np.abs(self.get_model_response(i, j, freqs)), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_deg(self, i : int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the phase in degrees of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, np.rad2deg(np.angle(self.network.s[:, i, j])), color='r', label='Samples')
        ax.plot(freqs, np.rad2deg(np.angle(self.get_model_response(i, j, freqs))), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (Degrees)')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_deg_unwrap(self, i : int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the unwrapped phase in degrees of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, np.rad2deg(np.unwrap(np.angle(self.network.s[:, i, j]))), color='r', label='Samples')
        ax.plot(freqs, np.rad2deg(np.unwrap(np.angle(self.get_model_response(i, j, freqs)))), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (Degrees)')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_re(self, i : int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the real part of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, np.real(self.network.s[:, i, j]), color='r', label='Samples')
        ax.plot(freqs, np.real(self.get_model_response(i, j, freqs)), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Real Part')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_im(self, i : int, j: int, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the imaginary part of the scattering parameter response :math:`S_{i+1,j+1}` in the fit.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        ax.scatter(self.network.f, np.imag(self.network.s[:, i, j]), color='r', label='Samples')
        ax.plot(freqs, np.imag(self.get_model_response(i, j, freqs)), color='k', label='Fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Imaginary Part')
        ax.legend(loc='best')
        ax.set_title('Response i={}, j={}'.format(i, j))
        return ax

    @check_plotting
    def plot_s_singular(self, freqs: Any = None, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the singular values of the vector fitted S-matrix in linear scale.

        Parameters
        ----------
        freqs : list of float or ndarray or None, optional
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network in
            :attr:`network` are used.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        if ax is None:
            ax = mplt.gca()

        # get system matrices of state-space representation
        A, B, C, D, E = self._get_ABCDE()

        n_ports = np.shape(D)[0]
        singvals = np.zeros((n_ports, len(freqs)))

        # calculate and save singular values for each frequency
        for i, f in enumerate(freqs):
            u, sigma, vh = np.linalg.svd(self._get_s_from_ABCDE(f, A, B, C, D, E))
            singvals[:, i] = sigma

        # plot the frequency response of each singular value
        for n in range(n_ports):
            ax.plot(freqs, singvals[n, :], label=r'$\sigma_{}$'.format(n + 1))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(loc='best')
        return ax

    @check_plotting
    def plot_pz(self, i: int, j: int, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots a pole-zero diagram of the fit of the model response :math:`H_{i+1,j+1}`.

        Parameters
        ----------
        i : int
            Row index of the response.

        j : int
            Column index of the response.

        ax : :class:`matplotlib.Axes` object or None
            matplotlib axes to draw on. If None, the current axes is fetched with :func:`gca()`.

        Returns
        -------
        :class:`matplotlib.Axes`
            matplotlib axes used for drawing. Either the passed :attr:`ax` argument or the one fetch from the current
            figure.
        """

        if ax is None:
            ax = mplt.gca()

        n_ports = int(np.sqrt(len(self.constant_coeff)))
        i_response = i * n_ports + j

        ax.scatter((np.real(self.poles), np.real(self.poles)),
                     (np.imag(self.poles), -1 * np.imag(self.poles)),
                     marker='x', label='Pole')
        ax.scatter((np.real(self.zeros[i_response]), np.real(self.zeros[i_response])),
                     (np.imag(self.zeros[i_response]), -1 * np.imag(self.zeros[i_response])),
                     marker='o', label='Zero')
        ax.set_xlabel('Re{s} (rad/s)')
        ax.set_ylabel('Im{s} (rad/s)')
        ax.legend(loc='best')
        return ax

    @check_plotting
    def plot_convergence(self, ax: mplt.Axes = None) -> mplt.Axes:
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

        if ax is None:
            ax = mplt.gca()

        ax.semilogy(np.arange(len(self.delta_max_history)) + 1, self.delta_max_history, color='darkblue')
        ax.set_xlabel('Iteration step')
        ax.set_ylabel('Max. relative change', color='darkblue')
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(self.d_res_history)) + 1, self.d_res_history, color='orangered')
        ax2.set_ylabel('Residue', color='orangered')
        return ax

    @check_plotting
    def plot_passivation(self, ax: mplt.Axes = None) -> mplt.Axes:
        """
        Plots the history of the greatest singular value during the iterative passivity enforcement process, which
        should eventually converge to a value slightly lower than 1.0 or stop after exceeding the maximum number of
        iterations specified in the class variable :attr:`history_max_sigma`.

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

        if ax is None:
            ax = mplt.gca()

        ax.plot(np.arange(len(self.history_max_sigma)) + 1, self.history_max_sigma)
        ax.set_xlabel('Iteration step')
        ax.set_ylabel('Max. singular value')
        return ax

    def write_spice_subcircuit_s(self, file: str) -> None:
        """
        Creates an equivalent N-port SPICE subcircuit based on its vector fitted S parameter responses.

        Parameters
        ----------
        file : str
            Path and filename including file extension (usually .sp) for the SPICE subcircuit file.

        Returns
        -------
        None

        Notes
        -----
        In the SPICE subcircuit, all ports will share a common reference node (global SPICE ground on node 0). The
        equivalent circuit uses linear dependent current sources on all ports, which are controlled by the currents
        through equivalent admittances modelling the parameters from a vector fit. This approach is based on [#]_.

        References
        ----------
        .. [#] G. Antonini, "SPICE Equivalent Circuits of Frequency-Domain Responses", IEEE Transactions on
            Electromagnetic Compatibility, vol. 45, no. 3, pp. 502-512, August 2003,
            DOI: https://doi.org/10.1109/TEMC.2003.815528
        """

        # list of subcircuits for the equivalent admittances
        subcircuits = []

        # provides a unique SPICE subcircuit identifier (X1, X2, X3, ...)
        def get_new_subckt_identifier():
            subcircuits.append('X{}'.format(len(subcircuits) + 1))
            return subcircuits[-1]

        # use engineering notation for the numbers in the SPICE file (1000 --> 1k)
        formatter = EngFormatter(sep="", places=3, usetex=False)
        # replace "micron" sign by "u" and "mega" sign by "meg"
        letters_dict = formatter.ENG_PREFIXES
        letters_dict.update({-6: 'u', 6: 'meg'})
        formatter.ENG_PREFIXES = letters_dict

        with open(file, 'w') as f:
            # write title line
            f.write('* EQUIVALENT CIRCUIT FOR VECTOR FITTED S-MATRIX\n')
            f.write('* Created using scikit-rf vectorFitting.py\n')
            f.write('*\n')

            # define the complete equivalent circuit as a subcircuit with one input node per port
            # those port nodes are labeled p1, p2, p3, ...
            # all ports share a common node for ground reference (node 0)
            str_input_nodes = ''
            for n in range(self.network.nports):
                str_input_nodes += 'p{} '.format(n + 1)

            f.write('.SUBCKT s_equivalent {}\n'.format(str_input_nodes))

            for n in range(self.network.nports):
                f.write('*\n')
                f.write('* port {}\n'.format(n + 1))
                # add port reference impedance z0 (has to be resistive, no imaginary part)
                f.write('R{} a{} 0 {}\n'.format(n + 1, n + 1, np.real(self.network.z0[0, n])))

                # add dummy voltage sources (V=0) to measure the input current
                f.write('V{} p{} a{} 0\n'.format(n + 1, n + 1, n + 1))

                # CCVS and VCVS driving the transfer admittances with a = V/2/sqrt(Z0) + I/2*sqrt(Z0)
                # In
                f.write('H{} nt{} nts{} V{} {}\n'.format(n + 1, n + 1, n + 1, n + 1, np.real(self.network.z0[0, n])))
                # Vn
                f.write('E{} nts{} 0 p{} 0 {}\n'.format(n + 1, n + 1, n + 1, 1))

                for j in range(self.network.nports):
                    f.write('* transfer network for s{}{}\n'.format(n + 1, j + 1))

                    # stacking order in VectorFitting class variables:
                    # s11, s12, s13, ..., s21, s22, s23, ...
                    i_response = n * self.network.nports + j

                    # add CCCS to generate the scattered current I_nj at port n
                    # control current is measured by the dummy voltage source at the transfer network Y_nj
                    # the scattered current is injected into the port (source positive connected to ground)
                    f.write('F{}{} 0 a{} V{}{} {}\n'.format(n + 1, j + 1, n + 1, n + 1, j + 1,
                                                            formatter(1 / np.real(self.network.z0[0, n]))))
                    f.write('F{}{}_inv a{} 0 V{}{}_inv {}\n'.format(n + 1, j + 1, n + 1, n + 1, j + 1,
                                                                    formatter(1 / np.real(self.network.z0[0, n]))))

                    # add dummy voltage source (V=0) in series with Y_nj to measure current through transfer admittance
                    f.write('V{}{} nt{} nt{}{} 0\n'.format(n + 1, j + 1, j + 1, n + 1, j + 1))
                    f.write('V{}{}_inv nt{} nt{}{}_inv 0\n'.format(n + 1, j + 1, j + 1, n + 1, j + 1))

                    # add corresponding transfer admittance Y_nj, which is modulating the control current
                    # the transfer admittance is a parallel circuit (sum) of individual admittances
                    f.write('* transfer admittances for S{}{}\n'.format(n + 1, j + 1))

                    # start with proportional and constant term of the model
                    # H(s) = d + s * e  model
                    # Y(s) = G + s * C  equivalent admittance
                    g = self.constant_coeff[i_response]
                    c = self.proportional_coeff[i_response]

                    # add R for constant term
                    if g < 0:
                        f.write('R{}{} nt{}{}_inv 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1, formatter(np.abs(1 / g))))
                    elif g > 0:
                        f.write('R{}{} nt{}{} 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1, formatter(1 / g)))

                    # add C for proportional term
                    if c < 0:
                        f.write('C{}{} nt{}{}_inv 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1, formatter(np.abs(c))))
                    elif c > 0:
                        f.write('C{}{} nt{}{} 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1, formatter(c)))

                    # add pairs of poles and zeros
                    for i_pole in range(len(self.poles)):
                        pole = self.poles[i_pole]
                        zero = self.zeros[i_response, i_pole]
                        node = get_new_subckt_identifier() + ' nt{}{}'.format(n + 1, j + 1)

                        if np.real(zero) < 0.0:
                            # multiplication with -1 required, otherwise the values for RLC would be negative
                            # this gets compensated by inverting the transfer current direction for this subcircuit
                            zero = -1 * zero
                            node += '_inv'

                        if np.imag(pole) == 0.0:
                            # real pole; add rl_admittance
                            l = 1 / np.real(zero)
                            r = -1 * np.real(pole) / np.real(zero)
                            f.write(node + ' 0 rl_admittance res={} ind={}\n'.format(formatter(r), formatter(l)))
                        else:
                            # complex pole of a conjugate pair; add rcl_vccs_admittance
                            l = 1 / (2 * np.real(zero))
                            b = -2 * (np.real(zero) * np.real(pole) + np.imag(zero) * np.imag(pole))
                            r = -1 * np.real(pole) / np.real(zero)
                            c = 2 * np.real(zero) / (np.abs(pole) ** 2)
                            gm_add = b * l * c
                            if gm_add < 0:
                                m = -1
                            else:
                                m = 1
                            f.write(node + ' 0 rcl_vccs_admittance res={} cap={} ind={} gm={} mult={}\n'.format(
                                formatter(r),
                                formatter(c),
                                formatter(l),
                                formatter(np.abs(gm_add)),
                                int(m)))

            f.write('.ENDS s_equivalent\n')

            f.write('*\n')

            # subcircuit for an active RCL+VCCS equivalent admittance Y(s) of a complex conjugated pole-zero pair H(s)
            # z = z' + j * z"
            # p = p' + j * p"
            # H(s)  = z / (s - p) + conj(z) / (s - conj(p))
            #       = (2 * z' * s - 2 * (z'p' + z"p")) / (s ** 2 - 2 * p' * s + |p| ** 2)
            # Y(S)  = (1 / L * s + b) / (s ** 2 + R / L * s + 1 / (L * C))
            f.write('.SUBCKT rcl_vccs_admittance n_pos n_neg res=1k cap=1n ind=100p gm=1m mult=1\n')
            f.write('L1 n_pos 1 {ind}\n')
            f.write('C1 1 2 {cap}\n')
            f.write('R1 2 n_neg {res}\n')
            f.write('G1 n_pos n_neg 1 2 {gm} m={mult}\n')
            f.write('.ENDS rcl_vccs_admittance\n')

            f.write('*\n')

            # subcircuit for a passive RL equivalent admittance Y(s) of a real pole-zero H(s)
            # H(s) = z / (s - p)
            # Y(s) = 1 / L / (s + s * R / L)
            f.write('.SUBCKT rl_admittance n_pos n_neg res=1k ind=100p\n')
            f.write('L1 n_pos 1 {ind}\n')
            f.write('R1 1 n_neg {res}\n')
            f.write('.ENDS rl_admittance\n')
