

'''
.. module:: skrf.media.freespace

========================================
freespace (:mod:`skrf.media.freespace`)
========================================

A Plane-wave in Freespace.
'''
from scipy.constants import  epsilon_0, mu_0
from numpy import real, imag
from .distributedCircuit import DistributedCircuit

class Freespace(DistributedCircuit):
    '''
    Represents a plane-wave in a homogeneous freespace, defined by
    the space's relative permativity and relative permeability.

    The field properties of space are related to a disctributed
    circuit transmission line model given in circuit theory by:

    ===============================  ==============================
    Circuit Property                 Field Property
    ===============================  ==============================
    distributed_capacitance          real(ep_0*ep_r)
    distributed_resistance           imag(ep_0*ep_r)
    distributed_inductance           real(mu_0*mu_r)
    distributed_conductance          imag(mu_0*mu_r)
    ===============================  ==============================

    .. ========================  =============  =================  ================================================
                           Circuit Property                 Field Property
            ---------------------------------------  -------------------------------------------------------------------
            Variable                  Symbol         Variable           Symbol
            ========================  =============  =================  ================================================
            distributed_capacitance   :math:`C^{'}`  real(ep_0*ep_r)    :math:`\\Re e \{\\epsilon_{0} \\epsilon_{r} \}`
            distributed_resistance    :math:`R^{'}`  imag(ep_0*ep_r)    :math:`\\Im m \{\\epsilon_{0} \\epsilon_{r} \}`
            distributed_inductance    :math:`I^{'}`  real(mu_0*mu_r)    :math:`\\Re e \{\\mu_{0} \\mu_{r} \}`
            distributed_conductance   :math:`G^{'}`  imag(mu_0*mu_r)    :math:`\\Im m \{\\mu_{0} \\mu_{r} \}`
            ========================  =============  =================  ================================================

    This class's inheritence is;
            :class:`~skrf.media.media.Media`->
            :class:`~skrf.media.distributedCircuit.DistributedCircuit`->
            :class:`~skrf.media.freespace.Freespace`

    '''
    def __init__(self, frequency,  ep_r=1+0j, mu_r=1+0j,  *args, **kwargs):
        '''
        Freespace initializer

        Parameters
        -----------
        frequency : :class:`~skrf.frequency.Frequency` object
                frequency band of this transmission line medium
        ep_r : number, array-like
                complex relative permativity
        mu_r : number, array-like
                possibly complex, relative permiability
        \*args, \*\*kwargs : arguments and keyword arguments

        Notes
        ------
        The distributed circuit parameters are related to a space's
        field properties by

        ===============================  ==============================
        Circuit Property                 Field Property
        ===============================  ==============================
        distributed_capacitance          real(ep_0*ep_r)
        distributed_resistance           :math:`\\omega  imag(ep_0*ep_r)
        distributed_inductance           real(mu_0*mu_r)
        distributed_conductance          :math:`\\omega  imag(mu_0*mu_r)
        ===============================  ==============================
        '''
        
        DistributedCircuit.__init__(self,\
                frequency = frequency, \
                C = real(epsilon_0*ep_r),\
                G = frequency.w *imag(epsilon_0*ep_r),\
                I = real(mu_0*mu_r),\
                R = frequency.w *imag(mu_0*mu_r),\
                *args, **kwargs
                )

    def __str__(self):
        f=self.frequency
        output =  \
                'Freespace  Media.  %i-%i %s.  %i points'%\
                (f.f_scaled[0],f.f_scaled[-1],f.unit, f.npoints)
        return output

    def __repr__(self):
        return self.__str__()
