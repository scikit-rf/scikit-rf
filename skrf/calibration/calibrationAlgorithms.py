
#       calibrationAlgorithms.py
#
#
#       Copyright 2010 alex arsenovic <arsenovic@virginia.edu>
#
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later versionpy.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
'''
.. module:: skrf.calibration.calibrationAlgorithms
====================================================================================
calibrationAlgorithms (:mod:`skrf.calibration.calibrationAlgorithms`)
====================================================================================




Contains calibrations algorithms and related functions, which are used
in the :class:`~skrf.calibration.calibration.Calibration` class.

Calibration Algorithms
-----------------------

.. autosummary::
   :toctree: generated/

        one_port
        one_port_nls
        two_port
        parameterized_self_calibration
        parameterized_self_calibration_nls

Supporting Functions
-----------------------
.. autosummary::
   :toctree: generated/

   unterminate_switch_terms
   abc_2_coefs_dict
   eight_term_2_one_port_coefs


'''

from copy import copy,deepcopy
import numpy as npy
from scipy import rand
from scipy.optimize import fmin_slsqp,fmin,leastsq # used for xds

from parametricStandard.parametricStandard import ParameterBoundsError
from ..mathFunctions import scalar2Complex, complex2Scalar

## Supporting Functions

def abc_2_coefs_dict(abc):
    '''
    converts an abc ndarry to a dictionarry containing the error
    coefficients.

    takes:
            abc : Nx3 numpy.ndarray, which holds the complex calibration
                    coefficients. the components of abc are
                            a[:] = abc[:,0]
                            b[:] = abc[:,1]
                            c[:] = abc[:,2],
                    a, b and c are related to the error network by
                            a = det(e) = e01*e10 - e00*e11
                            b = e00
                            c = e11
    returns:
            coefsDict: dictionary containing the following
                    'directivity':e00
                    'reflection tracking':e01e10
                    'source match':e11
    note:
            e00 = directivity error
            e10e01 = reflection tracking error
            e11 = source match error
    '''
    a,b,c = abc[:,0], abc[:,1],abc[:,2]
    e01e10 = a+b*c
    e00 = b
    e11 = c
    coefsDict = {\
            'directivity':e00,\
            'reflection tracking':e01e10, \
            'source match':e11\
            }
    return coefsDict

def eight_term_2_one_port_coefs(coefs):
    port1_coefs = {\
            'directivity':coefs['e00'],\
            'source match':coefs['e11'],\
            'reflection tracking':coefs['det_X']+ coefs['e00']*coefs['e11'],\
            }
    port2_coefs = {\
            'directivity':coefs['e33'],\
            'source match':coefs['e22'],\
            'reflection tracking':coefs['det_Y']+ coefs['e33']*coefs['e22'],\
            }
    return port1_coefs, port2_coefs

def guess_length_of_delay_short( aNtwk,tline):
    '''
    guess length of physical length of a Delay Short given by aNtwk

    takes:
            aNtwk: a skrf.ntwk type . (note: if this is a measurment
                    it needs to be normalized to the short plane
            tline: transmission line class of the medium. needed for the
                    calculation of propagation constant


    '''
    #TODO: re-write this and document better

    beta = npy.real(tline.beta())
    thetaM = npy.unwrap(npy.angle(-1*aNtwk.s).flatten())

    A = npy.vstack((-2*beta,npy.ones(len(beta)))).transpose()
    B = thetaM

    #print npy.linalg.lstsq(A, B)[1]/npy.dot(beta,beta)
    return npy.linalg.lstsq(A, B)[0][0]

def unterminate_switch_terms(two_port, gamma_f, gamma_r):
    '''
    unterminates switch terms from raw measurements.

    takes:
            two_port: the raw measurement, a 2-port Network type.
            gamma_f: the measured forward switch term, a 1-port Network type
            gamma_r: the measured reverse switch term, a 1-port Network type

    returns:
            un-terminated measurement, a 2-port Network type

    see:
            'Formulations of the Basic Vector Network Analyzer Error
            Model including Switch Terms' by Roger B. Marks
    '''
    unterminated = two_port.copy()

    # extract scattering matrices
    m, gamma_r, gamma_f = two_port.s, gamma_r.s, gamma_f.s
    u = m.copy()

    one = npy.ones(two_port.frequency.npoints)

    d = one - m[:,0,1]*m[:,1,0]*gamma_r[:,0,0]*gamma_f[:,0,0]
    u[:,0,0] = (m[:,0,0] - m[:,0,1]*m[:,1,0]*gamma_f[:,0,0])/(d)
    u[:,0,1] = (m[:,0,1] - m[:,0,0]*m[:,0,1]*gamma_r[:,0,0])/(d)
    u[:,1,0] = (m[:,1,0] - m[:,1,1]*m[:,1,0]*gamma_f[:,0,0])/(d)
    u[:,1,1] = (m[:,1,1] - m[:,0,1]*m[:,1,0]*gamma_r[:,0,0])/(d)

    unterminated.s = u
    return unterminated


## ONE PORT
def one_port(measured, ideals):
    '''
    Standard algorithm for a one port calibration.

    If more than three standards are supplied then a least square
    algorithm is applied.

    Parameters
    -----------
    measured : list of :class:`~....network.Network` objects or numpy.ndarray
            a list of the measured reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.
    ideals : list of :class:`~skrf.network.Network` objects or numpy.ndarray
            a list of the ideal reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.

    Returns
    -----------
    output : a dictionary
            output information from the calibration, the keys are
             * 'error coeffcients': dictionary containing standard error
               coefficients
             * 'residuals': a matrix of residuals from the least squared
               calculation. see numpy.linalg.lstsq() for more info


    Notes
    -----
            uses numpy.linalg.lstsq() for least squares calculation

    See Also
    ---------
            one_port_nls :  for a non-linear least square implementation
    '''
    #make  copies so list entities are not changed, when we typecast
    mList = measured[:]
    iList = ideals[:]

    numStds = len(mList)# find number of standards given, for dimensions
    numCoefs=3
    # try to access s-parameters, in case its a ntwk type, other wise
    # just keep on rollin
    try:
        for k in range(numStds):
            mList[k] = mList[k].s.reshape((-1,1))
            iList[k] = iList[k].s.reshape((-1,1))
    except:
        pass

    # ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
    fLength = len(mList[0])

    #initialize outputs
    abc = npy.zeros((fLength,numCoefs),dtype=complex)
    residuals =     npy.zeros((fLength,\
            npy.sign(numStds-numCoefs)),dtype=complex)
    parameter_variance = npy.zeros((fLength, 3,3),dtype=complex)
    measurement_variance = npy.zeros((fLength, 1),dtype=complex)
    # loop through frequencies and form m, a vectors and
    # the matrix M. where M =       i1, 1, i1*m1
    #                                                       i2, 1, i2*m2
    #                                                                       ...etc
    for f in range(fLength):
        #create  m, i, and 1 vectors
        one = npy.ones(shape=(numStds,1))
        m = npy.array([ mList[k][f] for k in range(numStds)]).reshape(-1,1)# m-vector at f
        i = npy.array([ iList[k][f] for k in range(numStds)]).reshape(-1,1)# i-vector at f

        # construct the matrix
        Q = npy.hstack([i, one, i*m])
        # calculate least squares
        abcTmp, residualsTmp = npy.linalg.lstsq(Q,m)[0:2]
        if numStds > 3:
            measurement_variance[f,:]= residualsTmp/(numStds-numCoefs)
            parameter_variance[f,:] = \
                    abs(measurement_variance[f,:])*\
                    npy.linalg.inv(npy.dot(Q.T,Q))

        # indicates singular value of matrix, but also same as having 3-standards
        #if len (residualsTmp ) == 0:
        #       raise ValueError( 'matrix has singular values, check complex distance of  standards')
        abc[f,:] = abcTmp.flatten()
        try:
            residuals[f,:] = residualsTmp
        except(ValueError):
            raise(ValueError('matrix has singular values. ensure standards are far enough away on smith chart'))


    # output is a dictionary of information
    output = {'error coefficients':abc_2_coefs_dict(abc), 'residuals':residuals, 'parameter variance':parameter_variance}

    return output

def one_port_nls (measured, ideals):
    '''
    one port non-linear least squares.

    Parameters
    -----------
    measured : list of :class:`~....network.Network` objects or numpy.ndarray
            a list of the measured reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.
    ideals : list of :class:`~skrf.network.Network` objects or numpy.ndarray
            a list of the ideal reflection coefficients. The elements
            of the list can  either a kxnxn numpy.ndarray, representing a
            s-matrix, or list of  1-port :class:`~skrf.network.Network`
            objects.

    Returns
    -----------
    output : a dictionary
            a dictionary containing the following keys:

            * 'error coeffcients': dictionary containing standard error
              coefficients

            * 'residuals': a matrix of residuals from the least squared
              calculation. see numpy.linalg.lstsq() for more info

            * 'cov_x': covariance matrix

    Notes
    ------
    Uses :func:`scipy.optmize.leastsq` for non-linear least squares
    calculation

    '''
    #make  copies so list entities are not changed, when we typecast
    mList = measured[:]
    iList = ideals[:]
    # find number of standards given, for dimensions
    numStds = len(mList)
    numCoefs=3
    # try to access s-parameters, in case its a ntwk type, other wise
    # just keep on rollin
    try:
        for k in range(numStds):
            mList[k] = mList[k].s.reshape((-1,1))
            iList[k] = iList[k].s.reshape((-1,1))
    except:
        pass

    # ASSERT: mList and aList are now kx1x1 matrices, where k in frequency
    fLength = len(mList[0])

    #initialize outputs
    abc = npy.zeros(shape=(fLength,numCoefs),dtype=complex)
    residuals = npy.zeros(shape=(fLength,numStds-numCoefs),dtype=complex)
    cov_x = []


    def residual_func(p, m,a):
        e00,e11,e0110 = scalar2Complex(p)
        m,a = scalar2Complex(m), scalar2Complex(a)
        m_i = ( e00 + e0110*a/(npy.ones(len(a))-e11*a))
        er = m-m_i
        er = er.real, er.imag#, npy.angle(er),
        return complex2Scalar(er)
    # loop through frequencies and form m, a vectors and
    # the matrix M. where M =       i1, 1, i1*m1
    #                                                       i2, 1, i2*m2
    #                                                                       ...etc
    for f in range(fLength):
        print f
        #create  m, i, and 1 vectors
        one = npy.ones(shape=(numStds,1))
        m = npy.array([ mList[k][f] for k in range(numStds)]).reshape(-1,1)# m-vector at f
        i = npy.array([ iList[k][f] for k in range(numStds)]).reshape(-1,1)# i-vector at f

        leastsq_output = leastsq(\
                func = residual_func, \
                x0 = [0, 0,0,0,1,0],\
                args = (complex2Scalar(m), complex2Scalar(i)),\
                full_output=True,\
                )
        e00,e11,e0110 = scalar2Complex(leastsq_output[0])
        abc[f,:] = [e0110-e00*e11, e00,e11]
        residualsTmp = (residual_func(leastsq_output[0], \
                complex2Scalar(m),complex2Scalar(i))**2).sum()
        s_sq = residualsTmp/(numStds*2 - numCoefs*2)
        cov_x.append(leastsq_output[1]* s_sq)
    # output is a dictionary of information
    output = {'error coefficients':abc_2_coefs_dict(abc), 'residuals':residuals, 'cov_x':npy.array(cov_x)}

    return output


## TWO PORT
def two_port(measured, ideals, switch_terms = None):
    '''
    Two port calibration based on the 8-term error model.

    Takes two
    ordered lists of measured and ideal responses. Optionally, switch
    terms [1]_ can be taken into account by passing a tuple containing the
    forward and reverse switch terms as 1-port Networks. This algorithm
    is based on the work in [2]_ .

    Parameters
    -----------
    measured : list of 2-port :class:`~skrf.network.Network` objects
            Raw measurements of the calibration standards. The order
            must align with the `ideals` parameter

    ideals : list of 2-port :class:`~skrf.network.Network` objects
            Predicted ideal response of the calibration standards.
            The order must align with `ideals` list
            measured: ordered list of measured networks. list elements

    switch_terms : tuple of :class:`~skrf.network.Network` objects
                    The two measured switch terms in the order
                    (forward, reverse).  This is only applicable in two-port
                    calibrations. See Roger Mark's paper on switch terms [1]_
                    for explanation of what they are.

    Returns
    ----------
    output : a dictionary
            output information, contains the following keys:
            * 'error coefficients':
            * 'error vector':
            * 'residuals':

    Notes
    ---------
    support for gathering switch terms on HP8510C  is in
    :mod:`skrf.virtualInstruments.vna`


    References
    -------------
    .. [1] Marks, Roger B.; , "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," ARFTG Conference Digest-Fall, 50th , vol.32, no., pp.115-126, Dec. 1997. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4119948&isnumber=4119931
    .. [2] Speciale, R.A.; , "A Generalization of the TSD Network-Analyzer Calibration Procedure, Covering n-Port Scattering-Parameter Measurements, Affected by Leakage Errors," Microwave Theory and Techniques, IEEE Transactions on , vol.25, no.12, pp. 1100- 1115, Dec 1977. URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1129282&isnumber=25047


    '''
    #make  copies so list entities are not changed, when we typecast
    mList = measured[:]
    iList = ideals[:]
    numStds = len(mList)# find number of standards given, for dimensions
    numCoefs = 7

    if len (mList) != len(iList):
        raise ValueError('Number of ideals must == number of measurements')

    if switch_terms is not None:
        mList = [unterminate_switch_terms(\
                                two_port = ntwk,\
                                gamma_f = switch_terms[0],\
                                gamma_r = switch_terms[1])\
                                for ntwk in mList]


    # try to access s-parameters, in case its a ntwk type, other wise
    # just keep on rollin
    try:
        for k in range(numStds):
            mList[k] = mList[k].s
            iList[k] = iList[k].s
    except:
        pass





    fLength = len(mList[0])
    #initialize outputs
    error_vector = npy.zeros(shape=(fLength,numCoefs),dtype=complex)
    residuals = npy.zeros(shape=(fLength,4*numStds-numCoefs),dtype=complex)
    Q = npy.zeros((numStds*4, 7),dtype=complex)
    M = npy.zeros((numStds*4, 1),dtype=complex)
    # loop through frequencies and form m, a vectors and
    # the matrix M. where M =       e00 + S11i
    #                                                       i2, 1, i2*m2
    #                                                                       ...etc
    for f in range(fLength):
        # loop through standards and fill matrix
        for k in range(numStds):
            m,i  = mList[k][f,:,:],iList[k][f,:,:] # 2x2 s-matrices
            Q[k*4:k*4+4,:] = npy.array([\
                    [ 1, i[0,0]*m[0,0], -i[0,0], 0 , i[1,0]*m[0,1],         0 ,     0        ],\
                    [ 0, i[0,1]*m[0,0], -i[0,1], 0 , i[1,1]*m[0,1],         0 ,  -m[0,1] ],\
                    [ 0, i[0,0]*m[1,0],     0,       0 , i[1,0]*m[1,1], -i[1,0],    0        ],\
                    [ 0, i[0,1]*m[1,0],     0,       1 , i[1,1]*m[1,1], -i[1,1], -m[1,1] ],\
                    ])
            #pdb.set_trace()
            M[k*4:k*4+4,:] = npy.array([\
                    [ m[0,0]],\
                    [       0       ],\
                    [ m[1,0]],\
                    [       0       ],\
                    ])

        # calculate least squares
        error_vector_at_f, residuals_at_f = npy.linalg.lstsq(Q,M)[0:2]
        #if len (residualsTmp )==0:
        #       raise ValueError( 'matrix has singular values, check standards')

        error_vector[f,:] = error_vector_at_f.flatten()
        residuals[f,:] = residuals_at_f

    # put the error vector into human readable dictionary
    error_coefficients = {\
            'e00':error_vector[:,0],\
            'e11':error_vector[:,1],\
            'det_X':error_vector[:,2],\
            'e33':error_vector[:,3]/error_vector[:,6],\
            'e22':error_vector[:,4]/error_vector[:,6],\
            'det_Y':error_vector[:,5]/error_vector[:,6],\
            'k':error_vector[:,6],\
            }

    # output is a dictionary of information
    output = {\
            'error coefficients':error_coefficients,\
            'error vector':error_vector, \
            'residuals':residuals\
            }

    return output


## SELF CALIBRATION
def parameterized_self_calibration(measured, ideals, showProgress=True,\
        **kwargs):
    '''
    An iterative, general self-calibration routine.

    A self calibration routine based off of residual error minimization
    which can take any mixture of parameterized standards.



    Parameters
    ------------
    measured : list of :class:`~....network.Network` objects
            a list of the measured networks
    ideals : list of :class:`~skrf.calibration.parametricStandard.parametricStandard.ParametricStandard` objects
            a list of the ideal networks
    showProgress : Boolean
            turn printing progress on/off
    **kwargs : key-word arguments
            passed to minimization algorithm (scipy.optimize.fmin)

    Returns
    -----------
    output : a dictionary
            a dictionary containing the following keys:

            * 'error_coefficients' : dictionary of error coefficients
            * 'residuals': residual matrix (shape depends on #stds)
            * 'parameter_vector_final': final results for parameter vector
            * 'mean_residual_list': the mean, magnitude of the residuals at each
                    iteration of calibration. this is the variable being minimized.

    See Also
    ----------
            :class:`parametricStandard` :  sub-module for more info on them
            parameterized_self_calibration_nls : similar algorithm, but uses
                    a non-linear least-squares estimator
    '''
    ideals_ps = ideals[:]
    #make copies so list entities are not changed
    measured = measured[:]
    if measured[0].number_of_ports ==1:
        cal_function = one_port
    elif measured[0].number_of_ports ==2:
        cal_function = two_port
    else:
        raise NotImplementedError('only 2 port supported')
    #note: ideals are passed by reference (not copied)

    # create the initial parameter vector
    parameter_vector = npy.array(())
    for a_ps in ideals_ps:
        parameter_vector = npy.append(parameter_vector, a_ps.parameter_array)
        #parameter_bounds = npy.append(parameter_bounds, a_ps.parameter_bounds)


    ideals = measured[:] #sloppy initalization, but this gets re-written by sub_cal
    mean_residual_list = []

    def sub_cal(parameter_vector, measured, ideals_ps):
        #TODO:  this function uses sloppy namespace, which limits portability

        # loop through the parameterized stds and assign the current
        # parameter vectors' elements to each std.
        p_index = 0 # index, of first element of current_ps in parameter vector
        for stdNum in range(len(ideals_ps)):
            current_ps = ideals_ps[stdNum]
            current_ps.parameter_array = \
                    parameter_vector[p_index:p_index+current_ps.number_of_parameters]
            try:
                ideals[stdNum]=current_ps.network
                ### HACK for parameterized standard sets.
                #ideals.append(current_ps.network)
            except (ParameterBoundsError):
                if showProgress:
                    print 'Bound Error:','==>',parameter_vector
                return  1e3* rand()
            p_index +=current_ps.number_of_parameters

        residues = cal_function(measured, ideals)['residuals']
        mean_residual_list.append((npy.mean(abs(residues))))

        if showProgress:
            print '%.3e'%mean_residual_list[-1],'==>',parameter_vector
        return mean_residual_list[-1]

    if showProgress:
        print ('| er |  ==>',[ k.parameter_keys for k in ideals_ps])
        print ('==================================================')
    parameter_vector_end = \
            fmin (sub_cal, parameter_vector,args=(measured,ideals_ps), **kwargs)

    output = cal_function(measured = measured, ideals=ideals)

    output.update( {\
    'parameter_vector_final':parameter_vector_end,\
    'mean_residual_list':mean_residual_list\
    })
    return output

def parameterized_self_calibration_nls(measured, ideals_ps, showProgress=True,\
        **kwargs):
    '''
    An iterative, general self-calibration routine.

    A self calibration routine based off of residual error minimization
    which can take any mixture of parameterized standards. Uses a
    non-linear least squares estimator to calculate the residuals.



    Parameters
    ------------
    measured : list of :class:`~....network.Network` objects
            a list of the measured networks
    ideals : list of :class:`~skrf.network.Network` objects
            a list of the ideal networks
    showProgress : Boolean
            turn printing progress on/off
    **kwargs : key-word arguments
            passed to minimization algorithm (scipy.optimize.fmin)

    Returns
    -----------
    output : a dictionary
            a dictionary containing the following keys:

            * 'error_coefficients' : dictionary of error coefficients
            * 'residuals': residual matrix (shape depends on #stds)
            * 'parameter_vector_final': final results for parameter vector
            * 'mean_residual_list': the mean, magnitude of the residuals at each
                    iteration of calibration. this is the variable being minimized.

    See Also
    ----------
            :class:`parametricStandard` :  sub-module for more info on them
            parameterized_self_calibration_nls : similar algorithm, but uses
                    a non-linear least-squares estimator
    '''
    #make copies so list entities are not changed
    measured = measured[:]
    if measured[0].number_of_ports ==1:
        cal_function = one_port
    elif measured[0].number_of_ports ==2:
        cal_function = two_port
    else:
        raise NotImplementedError('only 2 port supported')
    #note: ideals are passed by reference (not copied)

    # create the initial parameter vector
    parameter_vector = npy.array(())
    for a_ps in ideals_ps:
        parameter_vector = npy.append(parameter_vector, a_ps.parameter_array)
        #parameter_bounds = npy.append(parameter_bounds, a_ps.parameter_bounds)


    ideals = measured[:] #sloppy initalization, but this gets re-written by sub_cal
    mean_residual_list = []

    def sub_cal(parameter_vector, measured, ideals_ps):
        #TODO:  this function uses sloppy namespace, which limits portability

        # loop through the parameterized stds and assign the current
        # parameter vectors' elements to each std.
        p_index = 0 # index, of first element of current_ps in parameter vector
        for stdNum in range(len(ideals_ps)):
            current_ps = ideals_ps[stdNum]
            current_ps.parameter_array = \
                    parameter_vector[p_index:p_index+current_ps.number_of_parameters]
            ideals[stdNum]=current_ps.network
            p_index +=current_ps.number_of_parameters

        residues = cal_function(measured, ideals)['residuals']
        mean_residual_list.append(npy.mean(abs(residues)))

        if showProgress:
            print '%.3e'%mean_residual_list[-1],'==>',parameter_vector
        return mean_residual_list[-1]

    if showProgress:
        print ('| er |  ==>',[ k.parameter_keys for k in ideals_ps])
        print ('==================================================')
    parameter_vector_end = leastsq (\
            func=sub_cal,\
            x0=parameter_vector,\
            args=(measured,ideals_ps),\
            **kwargs)

    output = cal_function(measured = measured, ideals=ideals)

    output.update( {\
    'parameter_vector_final':parameter_vector_end,\
    'mean_residual_list':mean_residual_list\
    })
    return output

def parameterized_self_calibration_bounded(measured, ideals_ps, showProgress=True,\
        **kwargs):
    '''
    An iterative, general self-calibration routine which can take any
    mixture of parameterized standards. The correct parameter values
    are defined as the ones which minimize the mean residual error.



    takes:
            measured: list of Network types holding actual measurements
            ideals_ps: list of ParameterizedStandard types
            showProgress: turn printing progress on/off [boolean]
            **kwargs: passed to minimization algorithm (scipy.optimize.fmin)

    returns:
            a dictionary holding:
            'error_coefficients': dictionary of error coefficients
            'residuals': residual matrix (shape depends on #stds)
            'parameter_vector_final': final results for parameter vector
            'mean_residual_list': the mean, magnitude of the residuals at each
                    iteration of calibration. this is the variable being minimized.

    see  ParameterizedStandard for more info on them
    '''
    if len(measured) != len(ideals_ps):
        raise(IndexError('Number of ideals and measurements must be equal'))
    #make copies so list entities are not changed
    measured = measured[:]
    if measured[0].number_of_ports ==1:
        cal_function = one_port
    elif measured[0].number_of_ports ==2:
        cal_function = two_port
    else:
        raise NotImplementedError('only 2 port supported')
    #note: ideals are passed by reference (not copied)

    # create the initial parameter vector
    parameter_vector = npy.array(())
    parameter_bounds_list = []
    for a_ps in ideals_ps:
        parameter_vector = npy.append(parameter_vector, a_ps.parameter_array)
        if len(a_ps.parameter_bounds_array)!=0:
            parameter_bounds_list+=( a_ps.parameter_bounds_array)
    print parameter_bounds_list
    print parameter_vector

    ideals = measured[:] #sloppy initalization, but this gets re-written by sub_cal
    mean_residual_list = []

    def sub_cal(parameter_vector, measured, ideals_ps):
        #TODO:  this function uses sloppy namespace, which limits portability

        # loop through the parameterized stds and assign the current
        # parameter vectors' elements to each std.
        p_index = 0 # index, of first element of current_ps in parameter vector
        for stdNum in range(len(ideals_ps)):
            current_ps = ideals_ps[stdNum]
            current_ps.parameter_array = \
                    parameter_vector[p_index:p_index+current_ps.number_of_parameters]
            ideals[stdNum]=current_ps.network
            p_index +=current_ps.number_of_parameters

        residues = cal_function(measured, ideals)['residuals']
        mean_residual_list.append((npy.mean(abs(residues))))

        if showProgress:
            print '%.3e'%mean_residual_list[-1],'==>',parameter_vector
        return mean_residual_list[-1]

    if showProgress:
        print ('| er |  ==>',[ k.parameter_keys for k in ideals_ps])
        print ('==================================================')
    parameter_vector_end = \
            fmin_slsqp (\
                    func = sub_cal,\
                    x0 = parameter_vector,\
                    bounds = parameter_bounds_list,\
                    args=(measured,ideals_ps),\
                    **kwargs)

    output = cal_function(measured = measured, ideals=ideals)

    output.update( {\
    'parameter_vector_final':parameter_vector_end,\
    'mean_residual_list':mean_residual_list\
    })
    return output
