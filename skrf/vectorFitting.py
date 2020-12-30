import numpy as np
import os
import matplotlib.pyplot as mplt
from matplotlib.ticker import EngFormatter


class VectorFitting:
    """
    This is a Python implementation of the Vector Fitting algorithm [1]. The code is based on two improvements for
    relaxed pole relocation [2] and efficient (fast) solving [3]. See also the Vector Fitting website [4] for further
    information and download of the papers listed below. A Matlab implementation is also available there for reference.

    [1] B. Gustavsen, A. Semlyen, "Rational Approximation of Frequency Domain Responses by Vector Fitting", IEEE
        Transactions on Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999
    [2] B. Gustavsen, "Improving the Pole Relocating Properties of Vector Fitting", IEEE Transactions on Power Delivery,
        vol. 21, no. 3, pp. 1587-1592, July 2006
    [3] D. Deschrijver, M. Mrozowski, T. Dhaene, D. De Zutter, "Marcomodeling of Multiport Systems Using a Fast
        Implementation of the Vector Fitting Method", IEEE Microwave and Wireless Components Letters, vol. 18, no. 6,
        pp. 383-385, June 2008
    [4] https://www.sintef.no/projectweb/vectorfitting/
    """

    def __init__(self, network):
        """
        Creates a VectorFitting instance based on a supplied scikit-rf Network instance containing the frequency
        responses of the N-port.

        :param network:
            scikit-rf Network instance of the N-port to be fitted.
        """

        self.network = network
        self.initial_poles = None
        self.poles = None
        self.zeros = None
        self.proportional_coeff = None
        self.constant_coeff = None
        self.max_iterations = 100
        self.max_tol = 1e-6
        self.d_res_history = []
        self.delta_max_history = []

    def vectorfit(self, n_poles_real=2, n_poles_cmplx=2, init_pole_spacing='lin', parameter_type='S',
                  fit_constant=True, fit_proportional=True):
        """
        Main work routine performing the vector fit. The results will be stored in the class variables 'poles', 'zeros',
        'proportional_coeff' and 'constant_coeff'.

        :param n_poles_real:
            Number of initial real poles.

        :param n_poles_cmplx:
            Number of initial complex conjugate poles. Depends on the behaviour of the responses (e.g. resonances, ...).

        :param init_pole_spacing:
            Type of pole spacing for the initialisation. Either linear (lin) or logarithmic (log).

        :param parameter_type:
            Representation type of the frequency responses to be fitted. Either **scattering (S)**, **impedance (Z)**
            or **admittance (Y)**. As scikit-rf can currently only read S parameters from a Touchstone file, the fit
            should also be performed on the original S parameters. Otherwise, scikit-rf will convert the responses from
            S to Z or Y, which might work for the fit but can cause other issues.

        :param fit_constant:
            Include the constant term **d** in the fit.

        :param fit_proportional:
            Include the proportional term **e** in the fit.

        :return:
            None
        """

        # create initial poles and space them across the frequencies in the provided Touchstone file
        # use normalized frequencies during the iterations (seems to be more stable during least-squares fit)
        norm = np.average(self.network.f)
        freqs_norm = 2 * np.pi * self.network.f / norm

        fmin = np.amin(freqs_norm)
        fmax = np.amax(freqs_norm)
        weight_regular = 1.0
        if init_pole_spacing == 'log':
            pole_freqs = np.geomspace(fmin, fmax, n_poles_real + n_poles_cmplx)
        elif init_pole_spacing == 'lin':
            pole_freqs = np.linspace(fmin, fmax, n_poles_real + n_poles_cmplx)
        else:
            print('Invalid choice of initial pole spacing; proceeding with linear spacing')
            pole_freqs = np.linspace(fmin, fmax, n_poles_real + n_poles_cmplx)
        poles = []
        k_real = 0
        k_cmplx = 0
        for i, f in enumerate(pole_freqs):
            if i % 2 == 0 and k_real < n_poles_real:
                # add a real pole
                poles.append((-1 / 100 + 0j) * f)
                k_real += 1
            elif i % 2 == 1 and k_cmplx < n_poles_cmplx:
                # add a complex conjugate pole (store only the positive part)
                poles.append((-1 / 100 + 1j) * f)
                k_cmplx += 1
            elif k_real < n_poles_real:
                # add a real pole
                poles.append((-1 / 100 + 0j) * f)
                k_real += 1
            elif k_cmplx < n_poles_cmplx:
                # add a complex conjugate pole (store only the positive part)
                poles.append((-1 / 100 + 1j) * f)
                k_cmplx += 1
            else:
                # this should never occur
                print('error in pole init')
        poles = np.array(poles)

        # save initial poles (un-normalize first)
        self.initial_poles = poles * norm
        max_singular = 1

        print('### Starting pole relocation process.\n')

        # stack frequency responses as a single vector
        # stacking order:
        # s11, s12, s13, ..., s21, s22, s23, ...
        freq_responses = []
        for i in range(self.network.nports):
            for j in range(self.network.nports):
                if parameter_type == 'S':
                    freq_responses.append(self.network.s[:, i, j])
                elif parameter_type == 'Z':
                    freq_responses.append(self.network.z[:, i, j])
                elif parameter_type == 'Y':
                    freq_responses.append(self.network.y[:, i, j])
                else:
                    print('Invalid choice of matrix parameter type (S, Z, or Y); using S representation.')
                    freq_responses.append(self.network.s[:, i, j])
        freq_responses = np.array(freq_responses)

        # ITERATIVE FITTING OF POLES to the provided frequency responses
        # inital set of poles will be replaced with new poles after every iteration
        iterations = self.max_iterations
        converged = False
        while iterations > 0:
            print('Iteration {}'.format(self.max_iterations - iterations + 1))

            # generate coefficients of approximation function for each target frequency response
            # responses will be treated independently using QR decomposition
            # simplified coeff matrices of all responses will be stacked for least-squares solver
            A_matrix = []
            b_vector = []

            for freq_response in freq_responses:
                # calculate coefficients (row A_k in matrix) for each frequency sample s_k of the target response
                # row will be appended to submatrix A_sub of complete coeff matrix A_matrix
                # 2 rows per pole in result vector (1st for real part, 2nd for imaginary part)
                # --> 2 columns per pole in coeff matrix
                A_sub = []
                b_sub = []
                n_unused = 0

                for k, f_sample in enumerate(freqs_norm):
                    s_k = 1j * f_sample
                    A_k = []
                    n_unused = 0

                    # add coefficients for a pair of complex conjugate poles
                    # part 1: first sum of rational functions (residue variable c)
                    for pole in poles:
                        # seperate and stack real and imaginary part to preserve conjugacy of the pole pair
                        if np.imag(pole) == 0.0:
                            A_k.append(1 / (s_k - pole))
                            n_unused += 1
                        else:
                            # complex pole of a conjugated pair
                            A_k.append(1 / (s_k - pole) + 1 / (s_k - np.conjugate(pole)))       # real part of residue
                            A_k.append(1j / (s_k - pole) - 1j / (s_k - np.conjugate(pole)))     # imaginary part of residue
                            n_unused += 2

                    # part 2: constant (variable d) and proportional term (variable e)
                    if fit_constant:
                        A_k.append(1)
                        n_unused += 1
                    if fit_proportional:
                        A_k.append(s_k)
                        n_unused += 1

                    # part 3: second sum of rational functions (variable c_res)
                    for pole in poles:
                        # seperate and stack real and imaginary part to preserve conjugacy of the pole pair
                        if np.imag(pole) == 0.0:
                            A_k.append(-1 * freq_response[k] / (s_k - pole))
                        else:
                            # complex pole of a conjugated pair
                            A_k.append(-1 * freq_response[k] / (s_k - pole)
                                       - freq_response[k] / (s_k - np.conjugate(pole)))
                            A_k.append(-1j * freq_response[k] / (s_k - pole)
                                       + 1j * freq_response[k] / (s_k - np.conjugate(pole)))

                    # part 4: constant (variable d_res)
                    A_k.append(-1 * freq_response[k])

                    A_sub.append(np.sqrt(weight_regular) * np.real(A_k))
                    A_sub.append(np.sqrt(weight_regular) * np.imag(A_k))
                    b_sub.append(np.sqrt(weight_regular) * 0.0)
                    b_sub.append(np.sqrt(weight_regular) * 0.0)

                # QR decomposition
                Q, R = np.linalg.qr(A_sub, 'reduced')

                # only R22 is required to solve for c_res and d_res
                R22 = R[n_unused:, n_unused:]
                # similarly, only right half of Q is required
                Q2 = Q[:, n_unused:]

                if len(A_matrix) == 0:
                    A_matrix = R22
                    b_vector = np.matmul(np.transpose(Q2), b_sub)
                else:
                    A_matrix = np.vstack((A_matrix, R22))
                    b_vector = np.append(b_vector, np.matmul(np.transpose(Q2), b_sub))

                # add extra equation to avoid trivial solution
                # use weight=1 for all equations, except for this extra equation
                weight_extra = np.linalg.norm(weight_regular * freq_response) / len(freq_response)
                A_k = np.zeros(np.shape(A_matrix)[1])
                for k, f_sample in enumerate(freqs_norm):
                    s_k = 1j * f_sample
                    i = 0

                    # part 3: second sum of rational functions (variable c_res)
                    for pole in poles:
                        if np.imag(pole) == 0.0:
                            # real pole
                            A_k[i] += np.real(1 / (s_k - pole))
                            i += 1
                        else:
                            # complex pole of a conjugated pair
                            A_k[i] += np.real(1 / (s_k - pole) + 1 / (s_k - np.conjugate(pole)))
                            A_k[i+1] += np.real(1j / (s_k - pole) - 1j / (s_k - np.conjugate(pole)))
                            i += 2
                    # part 4: constant (d_res)
                    A_k[i] += 1

                A_matrix = np.vstack((A_matrix, np.sqrt(weight_extra) * A_k))
                b_vector = np.append(b_vector, np.sqrt(weight_extra) * len(freqs_norm))

            print('A_matrix: condition number = {}'.format(np.linalg.cond(A_matrix)))

            # solve least squares for real parts
            x, residuals, rank, singular_vals = np.linalg.lstsq(A_matrix, b_vector, rcond=-1)

            # assemble individual result vectors from single LS result x
            c_res = x[:-1]
            d_res = x[-1]

            # check if d_res is suited for zeros calculation
            tol_res = 1e-8
            if np.abs(d_res) < tol_res:
                # d_res is too small, discard solution and proceed the |d_res| = tol_res
                d_res = tol_res * (d_res / np.abs(d_res))
                print('Replacing d_res solution')

            self.d_res_history.append(d_res)
            print('d_res = {}'.format(d_res))

            # build test matrix H, which will hold the new poles as Eigenvalues
            A_matrix = np.zeros((len(c_res), len(c_res)))
            i = 0
            for pole in poles:
                # fill diagonal with previous poles
                if np.imag(pole) == 0.0:
                    # one row for a real pole
                    A_matrix[i, i] = np.real(pole)
                    A_matrix[i] -= c_res / d_res
                    i += 1
                else:
                    A_matrix[i] -= 2 * c_res / d_res
                    # two rows for a complex pole of a conjugated pair
                    A_matrix[i, i] = np.real(pole)
                    A_matrix[i, i + 1] = np.imag(pole)
                    A_matrix[i + 1, i] = -1 * np.imag(pole)
                    A_matrix[i + 1, i + 1] = np.real(pole)
                    i += 2
            poles_new = np.linalg.eigvals(A_matrix)

            # replace poles for next iteration
            poles = []
            for k, pole in enumerate(poles_new):
                # flip real part of unstable poles (real part needs to be negative for stability)
                if np.real(pole) > 0.0:
                    pole = -1 * np.real(pole) + 1j * np.imag(pole)
                if np.imag(pole) >= 0.0:
                    # complex poles need to come in complex conjugate pairs; append only the positive part
                    poles.append(pole)

            poles = np.sort_complex(poles)

            # calculate relative changes in the singular values; stop iteration loop once poles have converged
            new_max_singular = np.amax(singular_vals)
            delta_max = np.abs(1 - new_max_singular / max_singular)
            self.delta_max_history.append(delta_max)
            print('Delta_max = {}'.format(delta_max))
            print()
            max_singular = new_max_singular

            stop = False
            if delta_max < self.max_tol:
                if converged:
                    # is really converged, finish
                    print('Pole relocation process converged after {} iterations.'.format(self.max_iterations - iterations + 1))
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
                if converged and stop is False:
                    print('Reached tolerance only after max. number of iterations (N_max = {}). '
                          'Results might not have converged properly.'.format(self.max_iterations))
                else:
                    print('Reached maximum number of iterations (N_max = {}). '
                          'Results did not converge.'.format(self.max_iterations))

            if stop:
                iterations = 0

        # ITERATIONS DONE
        poles = np.array(poles)

        print('Initial poles before relocation:')
        print(self.initial_poles)
        print()

        print('Final poles:')
        print(poles * norm)
        print()

        print('\n### Starting zeros calculation process.\n')

        # finally, solve for the residues with the previously calculated poles
        zeros = []
        constant_coeff = []
        proportional_coeff = []

        for freq_response in freq_responses:
            # calculate coefficients (row A_k in matrix) for each frequency sample s_k of the target response
            # row will be appended to submatrix A_sub of complete coeff matrix A_matrix
            # 2 rows per pole in result vector (1st for real part, 2nd for imaginary part)
            # --> 2 columns per pole in coeff matrix
            A_matrix = []
            b_vector = []

            for k, f_sample in enumerate(freqs_norm):
                s_k = 1j * f_sample
                A_k = []
                # add coefficients for a pair of complex conjugate poles
                # part 1: first sum of rational functions (residue variable c)
                for pole in poles:
                    # separate and stack real and imaginary part to preserve conjugacy of the pole pair
                    if np.imag(pole) == 0.0:
                        A_k.append(1 / (s_k - pole))
                    else:
                        A_k.append(1 / (s_k - pole) + 1 / (s_k - np.conjugate(pole)))       # real part of residue
                        A_k.append(1j / (s_k - pole) - 1j / (s_k - np.conjugate(pole)))     # imaginary part of residue

                # part 2: constant (variable d) and proportional term (variable e)
                if fit_constant:
                    A_k.append(1.0)
                if fit_proportional:
                    A_k.append(s_k)

                A_matrix.append(np.array(A_k))
                b_vector.append(np.array(freq_response[k]))

            A_matrix = np.vstack((np.real(A_matrix), np.imag(A_matrix)))
            b_vector = np.append(np.real(b_vector), np.imag(b_vector))

            print('A_matrix: condition number = {}'.format(np.linalg.cond(A_matrix)))

            # solve least squares and obtain results as stack of real part vector and imaginary part vector
            x, residuals, rank, singular_vals = np.linalg.lstsq(A_matrix, b_vector, rcond=-1)

            i = 0
            zeros_response = []
            for pole in poles:
                if np.imag(pole) == 0.0:
                    zeros_response.append(x[i] + 0j)
                    i += 1
                else:
                    zeros_response.append(x[i] + 1j * x[i + 1])
                    i += 2

            zeros.append(zeros_response)
            if fit_constant:
                constant_coeff.append(x[-2])
            else:
                constant_coeff.append(0.0)
            if fit_proportional:
                proportional_coeff.append(x[-1])
            else:
                proportional_coeff.append(0.0)

        # save poles, zeros, d, e in actual frequencies (un-normalized)
        self.poles = poles * norm
        self.zeros = np.array(zeros) * norm
        self.constant_coeff = np.array(constant_coeff)
        self.proportional_coeff = np.array(proportional_coeff) / norm

        print('\n### Vector fitting finished.\n')

    def export_npz(self, path):
        """
        Exports the poles vector, zeros matrix, proportional coefficients vector and constants vector as a NumPy .npz
        array.

        :param path:
            Target path for the export.

        :return:
            None
        """

        if self.poles is None:
            print('Nothing to export; Poles have not been fitted.')
            return
        if self.zeros is None:
            print('Nothing to export; Zeros have not been fitted.')
            return
        if self.proportional_coeff is None:
            print('Nothing to export; Proportional coefficients have not been fitted.')
            return
        if self.constant_coeff is None:
            print('Nothing to export; Constants have not been fitted.')
            return

        filename = self.network.name

        print('Exporting results as pickled NumPy array to {}'.format(path))
        np.savez(os.path.join(path, 'coefficients_{}'.format(filename)),
                 poles=self.poles, zeros=self.zeros, proportionals=self.proportional_coeff,
                 constants=self.constant_coeff)

    def import_npz(self, file):
        """
        Imports all model parameters (poles, zeros, proportionals, constants) from a provided NumPy .npz file.

        :param file:
            NumPy .npz file containing all parameters as properly labeled arrays with the correct lengths (matching the
            number of ports of the scikit-rf Network in self.network).

        :return:
            None
        """

        with np.load(file) as data:
            poles = data['poles']
            zeros = data['zeros']
            proportional_coeff = data['proportionals']
            constant_coeff = data['constants']

            if np.alen(zeros) == self.network.nports and \
                    np.alen(proportional_coeff) == self.network.nports and \
                    np.alen(constant_coeff) == self.network.nports:
                self.poles = poles
                self.zeros = zeros
                self.proportional_coeff = proportional_coeff
                self.constant_coeff = constant_coeff
            else:
                print('IMPORT ERROR: Length of the provided parameters does not match the network size.')

    def get_model_response(self, i, j, freqs=None):
        """
        Returns the frequency response of the fitted model.

        :param i:
            Row index of the response.

        :param j:
            Column index of the response.

        :param freqs:
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network are used.

        :return:
            Model response at the provided frequencies (complex valued list).
        """

        if self.poles is None:
            print('Returning zero; Poles have not been fitted.')
            return np.zeros_like(freqs)
        if self.zeros is None:
            print('Returning zero; Zeros have not been fitted.')
            return np.zeros_like(freqs)
        if self.proportional_coeff is None:
            print('Returning zero; Proportional coefficients have not been fitted.')
            return np.zeros_like(freqs)
        if self.constant_coeff is None:
            print('Returning zero; Constants have not been fitted.')
            return np.zeros_like(freqs)
        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        s = 2j * np.pi * np.array(freqs)
        i_response = i * self.network.nports + j
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

    def plot_fit_linear(self, i, j, freqs=None):
        """
        Plots the magnitude in dB of the response **S_i,j** in the fit.

        :param i:
            Row index of the response.

        :param j:
            Column index of the response.

        :param freqs:
            List of frequencies for the response plot. If None, the sample frequencies of the fitted network are used.

        :return:
            None
        """

        if freqs is None:
            freqs = np.linspace(np.amin(self.network.f), np.amax(self.network.f), 1000)

        mplt.figure()
        #mplt.scatter(self.network.f, 20 * np.log10(np.abs(self.network.s[:, i, j])), color='r', label='Samples')
        mplt.scatter(self.network.f, np.abs(self.network.s[:, i, j]), color='r', label='Samples')
        #mplt.plot(freqs, 20 * np.log10(np.abs(self.get_model_response(i, j, freqs))), color='k', label='Fit')
        mplt.plot(freqs, np.abs(self.get_model_response(i, j, freqs)), color='k', label='Fit')
        mplt.xlabel('Frequency (Hz)')
        #mplt.ylabel('Magnitude (dB)')
        mplt.ylabel('Magnitude')
        mplt.legend(loc='best')
        mplt.title('i={}, j={}'.format(i, j))
        mplt.tight_layout()
        mplt.show()

    def plot_pz(self, i, j):
        """
        Plots a pole-zero diagram of the fit of the response **S_i,j**.

        :param i:
            Row index of the response.

        :param j:
            Column index of the response.

        :return:
            None
        """

        i_response = i * self.network.nports + j
        mplt.figure()
        mplt.scatter((np.real(self.poles), np.real(self.poles)),
                     (np.imag(self.poles), -1 * np.imag(self.poles)),
                     marker='x', label='Pole')
        mplt.scatter((np.real(self.zeros[i_response]), np.real(self.zeros[i_response])),
                     (np.imag(self.zeros[i_response]), -1 * np.imag(self.zeros[i_response])),
                     marker='o', label='Zero')
        mplt.xlabel('Re{s} (rad/s)')
        mplt.ylabel('Im{s} (rad/s)')
        mplt.legend(loc='best')
        mplt.tight_layout()
        mplt.show()

    def plot_convergence(self):
        """
            Plots the history of the model parameter **d_res** during the iterative pole relocation process of the
            vector fitting, which should eventually converge to a fixed value. Additionally, the relative change of the
            maximum singular value of the coefficient matrix **A** are plotted, which serve as a convergence indicator.

        :return:
            None
        """
        mplt.figure()
        mplt.semilogy(np.arange(np.alen(self.delta_max_history)) + 1, self.delta_max_history, color='darkblue')
        mplt.xlabel('Iteration step')
        mplt.ylabel('Max. relative change', color='darkblue')
        ax2 = mplt.twinx()
        ax2.plot(np.arange(np.alen(self.d_res_history)) + 1, self.d_res_history, color='orangered')
        ax2.set_ylabel('Residue', color='orangered')
        mplt.tight_layout()
        mplt.show()

    def write_spice_subckt_s(self, file):
        """
        Creates an equivalent N-port SPICE subcircuit based on its vector fitted S parameter responses. In the SPICE
        subcircuit, all ports will share a common reference node (global SPICE ground on node 0). The equivalent circuit
        uses linear dependent current sources on all ports, which are controlled by the currents through equivalent
        admittances modelling the parameters from a vector fit. This approach is based on [5].

        [5] G. Antonini, "SPICE Equivalent Circuits of Frequency-Domain Responses", IEEE Transactions on Electromagnetic
            Compatibility, vol. 45, no. 3, pp. 502-512, August 2003

        :param file:
            String of the path and filename including file extension (usually .sp) for the SPICE file.

        :return:
            None
        """

        # list of subcircuits for the equivalent admittances
        subcircuits = []

        # provides a unique SPICE subcircuit identifier (X1, X2, X3, ...)
        def get_new_subckt_identifier():
            subcircuits.append('X{}'.format(len(subcircuits) + 1))
            return subcircuits[-1]

        # use engineering notation for the numbers in the SPICE file (1000 --> 1k)
        self.formatter = EngFormatter(sep="", places=3)
        # replace "micron" sign by "u" and "mega" sign by "meg"
        letters_dict = self.formatter.ENG_PREFIXES
        letters_dict.update({-6: 'u', 6: 'meg'})
        self.formatter.ENG_PREFIXES = letters_dict

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
                f.write('R{} a{} 0 {}\n'.format(n + 1, n + 1, self.network.z0[0, n]))

                # add dummy voltage sources (V=0) to measure the input current
                f.write('V{} p{} a{} 0\n'.format(n + 1, n + 1, n + 1))

                # CCVS and VCVS driving the transfer admittances with a = V/2/sqrt(Z0) + I/2*sqrt(Z0)
                # In
                f.write('H{} nt{} nts{} V{} {}\n'.format(n + 1, n + 1, n + 1, n + 1, self.network.z0[0, n]))
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
                    f.write('F{}{} 0 a{} V{}{} {}\n'.format(n + 1, j + 1, n + 1, n + 1, j + 1, 1 / self.network.z0[0, n]))
                    f.write('F{}{}_inv a{} 0 V{}{}_inv {}\n'.format(n + 1, j + 1, n + 1, n + 1, j + 1, 1 / self.network.z0[0, n]))

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
                        f.write('R{}{} nt{}{}_inv 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1,
                                                                 self.formatter(np.abs(1 / g))))
                    elif g > 0:
                        f.write('R{}{} nt{}{} 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1,
                                                             self.formatter(1 / g)))

                    # add C for proportional term
                    if c < 0:
                        f.write('C{}{} nt{}{}_inv 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1,
                                                                 self.formatter(np.abs(c))))
                    elif c > 0:
                        f.write('C{}{} nt{}{} 0 {}\n'.format(n + 1, j + 1, n + 1, j + 1,
                                                             self.formatter(c)))

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
                            f.write(node + ' 0 rl_admittance res={} ind={}\n'.format(
                                self.formatter(r),
                                self.formatter(l)))
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
                                self.formatter(r),
                                self.formatter(c),
                                self.formatter(l),
                                self.formatter(np.abs(gm_add)),
                                int(m)))

            f.write('.ENDS s_equivalent\n')

            f.write('*\n')

            # subcircuit for an active RCL+VCCS equivalent admittance Y(s) of a complex conjugated pole-zero pair H(s)
            # H(s) = z / (s - p) + conj(z) / (s - conj(p))
            # Y(S) =
            f.write('.SUBCKT rcl_vccs_admittance n_pos n_neg res=1k cap=1n ind=100p gm=1m mult=1\n')
            f.write('L1 n_pos 1 {ind}\n')
            f.write('C1 1 2 {cap}\n')
            f.write('R1 2 n_neg {res}\n')
            f.write('G1 n_pos n_neg 1 2 {gm} m={mult}\n')
            f.write('.ENDS rcl_admittance\n')

            f.write('*\n')

            # subcircuit for a passive RL equivalent admittance Y(s) of a real pole-zero H(s)
            # H(s) = z / (s - p)
            # Y(s) =
            f.write('.SUBCKT rl_admittance n_pos n_neg res=1k ind=100p\n')
            f.write('L1 n_pos 1 {ind}\n')
            f.write('R1 1 n_neg {res}\n')
            f.write('.ENDS rl_admittance\n')
