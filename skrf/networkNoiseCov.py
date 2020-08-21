import numpy as npy
from copy import deepcopy as copy

from .util import network_array
from .constants import ZERO, K_BOLTZMANN, h_PLANK

class NetworkNoiseCov(object):

    COVARIANCE_FORMS = ['s', 't', 'z', 'y', 'a']

    def __init__(self, mat_vec, form='s', z0 = 50, T0=290):

        self._validate_mat_vec_form(mat_vec)
        self._validate_form(form)
        self._validate_z0(z0)

        self._mat_vec = mat_vec
        self._form = form
        self._z0 = z0
        self._T0 = T0
        self._k_norm = 1/npy.sqrt(2*self._z0)

        # dictionaries of transforms. Transforming is carried out during getter and setter operations 
        self.transform_to_s = {'s': self._do_nothing, 't': self._ct2cs, 'z': self._cz2cs, 'y': self._cy2cs,'a': self._ca2cs }
        self.transform_to_t = {'s': self._cs2ct, 't': self._do_nothing, 'z': self._cz2ct, 'y': self._cy2ct,'a': self._ca2ct }
        self.transform_to_z = {'s': self._cs2cz, 't': self._ct2cz, 'z': self._do_nothing, 'y': self._cy2cz,'a': self._ca2cz }
        self.transform_to_y = {'s': self._cs2cy, 't': self._ct2cy, 'z': self._cz2cy, 'y': self._do_nothing,'a': self._ca2cy }
        self.transform_to_a = {'s': self._cs2ca, 't': self._ct2ca, 'z': self._cz2ca, 'y': self._cy2ca,'a': self._do_nothing }

    @classmethod
    def Tnoise(cls,f,T0):

        #This is the correct blackbody noise that accounts for the quantum limit to thermal noise for low values of T0 as well as
        #the upper noise limit for frequencies that exceed 200 GHz. 
        #Insert reference here - MBG

        X = (h_PLANK*f)/(2*K_BOLTZMANN*T0)
        Tn = ((h_PLANK*f)/(2*K_BOLTZMANN))*(1/npy.tanh(X))

        return Tn


    @classmethod
    def from_passive_z(cls, z, z0=50, T0=290):

        Tn = cls.Tnoise(f,T0)
        Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(s)[1],npy.shape(s)[2]))

        cov = 4.*K_BOLTZMANN*Tn_mat*npy.real(z)
        return cls(cov, form='z', z0=z0, T0=T0)

    @classmethod
    def from_passive_y(cls, y, f, z0=50, T0=290):

        Tn = cls.Tnoise(f,T0)
        Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(s)[1],npy.shape(s)[2]))

        cov = 4.*K_BOLTZMANN*Tn_mat*npy.real(y)
        return cls(cov, form='y', z0=z0, T0=T0)

    @classmethod
    def from_passive_s(cls, s, f, z0=50, T0=290):

        Tn = cls.Tnoise(f,T0)
        Tn_mat = npy.tile(Tn[:,None,None], (1,npy.shape(s)[1],npy.shape(s)[2]))
  
        #ovec = npy.ones(s.shape[0])
        #zvec = npy.zeros(s.shape[0])
        SM =  npy.matmul(s, npy.conjugate(s.swapaxes(1, 2)))
        I_2D = npy.identity(npy.shape(s)[1])
        I = npy.repeat(I_2D[npy.newaxis,:, :], npy.shape(s)[0], axis=0)
        #I = network_array([[ovec, zvec],[zvec, ovec]])
        cov = K_BOLTZMANN*Tn_mat*(I - SM)/2
        return cls(cov, form='s', z0=z0, T0=T0)


    def copy(self):
        '''
        Returns a copy of this NetworkNoiseCov

        '''
        n = NetworkNoiseCov(mat_vec=self._mat_vec, form = self._form, z0 = self._z0)
        return n

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, value):
        if value not in self.COVARIANCE_FORMS:
            raise ValueError("form must be one of \'s\', \'t\', \'z\', \'y\', \'a\'" )
        self._form = value

    @property
    def mat_vec(self):
        return self._mat_vec

    @mat_vec.setter
    def mat_vec(self, value):
        if value.shape != self._mat_vec.shape:
            raise ValueError("mat_vec " + str(value.shape) +  " to " + str(self._mat_vec.shape) + " incompatible" )
        self._mat_vec = value

    @property
    def cc(self):
        return self.mat_vec


    def get_cs(self, S):
        return self.transform_to_s[self.form](self._mat_vec, S)

    def get_ct(self, T):
        return self.transform_to_t[self.form](self._mat_vec, T)

    def get_cz(self, Z):
        return self.transform_to_t[self.form](self._mat_vec, Z)

    def get_cy(self, Y):
        return self.transform_to_y[self.form](self._mat_vec, Y)

    def get_ca(self, A):
        return self.transform_to_a[self.form](self._mat_vec, A)

    @property
    def y_opt(self):
        self._validate_only_if_ca()
        ca = self.mat_vec
        return (npy.sqrt(ca[:,1,1]/ca[:,0,0] - npy.square(npy.imag(ca[:,0,1]/ca[:,0,0])))
          + 1.j*npy.imag(ca[:,0,1]/ca[:,0,0]))

    @property
    def z_opt(self):
        return 1./self.y_opt

    @property
    def nfmin(self):
        self._validate_only_if_ca()
        ca = self.mat_vec
        return npy.real(1. + (ca[:,0,1] + ca[:,0,0] * npy.conj(self.y_opt))/(2.*K_BOLTZMANN*self._T0))

    @property
    def nfmin_db(self):
        return mf.complex_2_db10(self.nfmin)

    @property
    def rn(self):
        self._validate_only_if_ca()
        ca11 = self.mat_vec[:,0,0]
        return npy.real(ca11/(4.*K_BOLTZMANN*self._T0))

    def nf(self, z):
     
        z0 = self.z0
        y_opt = self.y_opt
        fmin = self.nfmin
        rn = self.rn

        ys = 1./z
        gs = npy.real(ys)
        return fmin + rn/gs * npy.square(npy.absolute(ys - y_opt))

    def _validate_only_if_ca(self):
        if self.form != 'a':
            raise ValueError("Noise parameters can only be extracted from an ABCD form NetworksNoiseCov object at this time")

    def _validate_mat_vec_form(self, mat_vec):
        """make sure the shape is correct
        """
        pass

    def _validate_form(self, form):
        if form not in self.COVARIANCE_FORMS:
            raise ValueError("form must be one of 's', 't', 'z', 'y', 'a'" )

    def _validate_z0(self, z0):
        """For right now z0 needs to be a real constant
        """
        pass

    ## Covariance form conversions
    def _do_nothing(self, value, M):
        return value

    ## S to other
    def _cs2ct(self, mat, T):
        t12 = T[:, 0, 1]
        t22 = T[:, 1, 1]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -t12],[zvec, -t22]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 't'
        return n

    def _cs2cz(self, mat, Z):
        Zn = Z/self._z0
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (Zn + I)/self._k_norm
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _cs2cy(self, mat, Y):
        raise NotImplemented()

    def _cs2ca(self, mat, A):
        Z = self._z2a(A)
        cz = self._cs2cz(mat, Z)
        ca = self._cz2ca(cz.mat_vec, A)
        ca.form = 'a'
        return ca

    ## T to other
    def _ct2cs(self, mat, S):
        s11 = S[:,0,0]
        s21 = S[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -s11],[zvec, -s21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _ct2cz(self, mat, Z):
        raise NotImplemented()

    def _ct2cy(self, mat, Y):
        raise NotImplemented()

    def _ct2ca(self, mat, A):
        Z = self._z2a(A) # equivalent to a2z
        S = self._z2s(Z) # 
        cs = self._ct2cs(mat, S)
        cz = self._cs2cz(cs.mat_vec, Z)
        ca = self._cz2ca(cz.mat_vec, A)
        ca.form = 'a'
        return ca

    ## Z to other
    def _cz2cs(self, mat, S):
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (I - S)*self._k_norm/2
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _cz2ct(self, mat, T):
        S = self._t2s(T) 
        cs = self._cz2cs(mat, S)
        ct = self._cs2ct(cs.mat_vec, T)
        ct.form = 't'
        return ct

    def _cz2cy(self, mat, Y):
        Tm = Z
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'y'
        return n

    def _cz2ca(self, mat, A):
        a11 = A[:,0,0]
        a21 = A[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -a11],[zvec, -a21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'a'
        return n

    ## Y to other
    def _cy2cs(self, mat, S):
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        I = network_array([[ovec, zvec],[zvec, ovec]])
        Tm = (I + S)*self._k_norm*self._z0/2
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 's'
        return n

    def _cy2ct(self, mat, T):
        S = self._t2s(T) 
        cs = self._cy2cs(mat, S)
        ct = self._cs2ct(cs.mat_vec, T)
        ct.form = 't'
        return ct

    def _cy2cz(self, mat, Z):
        Tm = Z
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _cy2ca(self, mat, A):
        a12 = A[:, 0, 1]
        a22 = A[:, 1, 1]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[zvec, a12],[ovec, a22]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'a'
        return n

    ## A to other
    def _ca2cs(self, mat, S):
        raise NotImplemented()

    def _ca2ct(self, mat, T):
        raise NotImplemented()

    def _ca2cz(self, mat, Z):
        z11 = Z[:,0,0]
        z21 = Z[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[ovec, -z11],[zvec, -z21]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'z'
        return n

    def _ca2cy(self, mat, Y):
        y11 = Y[:,0,0]
        y21 = Y[:,1,0]
        ovec = npy.ones(mat.shape[0])
        zvec = npy.zeros(mat.shape[0])
        Tm = network_array([[-y11, ovec],[-y21,zvec]])
        n = self.copy()
        n.mat_vec = npy.matmul(Tm, npy.matmul(mat, npy.conjugate(Tm.swapaxes(1, 2))))
        n.form = 'y'
        return n

    # Need these until I fill in the rest of the transforms above
    def _z2a(self,z):
        '''
        z2a = a2z
        '''
        abcd = npy.array([
            [z[:, 0, 0] / z[:, 1, 0],
            1. / z[:, 1, 0]],
            [(z[:, 0, 0] * z[:, 1, 1] - z[:, 1, 0] * z[:, 0, 1]) / z[:, 1, 0],
            z[:, 1, 1] / z[:, 1, 0]],
        ]).transpose()
        return abcd

    
    def _z2s(self, z, z0=50, s_def='power'):
        """
        convert impedance parameters [1]_ to scattering parameters [2]_

        For power-waves, Eq.(18) from [3]:

        .. math::
            S = F (Z – G^*) (Z + G)^{-1} F^{-1}

        where :math:`G = diag([Z_0])` and :math:`F = diag([1/2\\sqrt{|Re(Z_0)|}])`  
            
        For pseudo-waves, Eq.(73) from [4]:

        .. math::
            S = U (Z - G) (Z + G)^{-1}  U^{-1}

        where :math:`U = \\sqrt{Re(Z_0)}/|Z_0|`


        Parameters
        ------------
        z : complex array-like
            impedance parameters
        z0 : complex array-like or number
            port impedances
        s_def : str -> s_def : ['power','pseudo']
            Scattering parameter definition : 'power' for power-waves definition [3], 
            'pseudo' for pseudo-waves definition [4]. Default is 'power'.

        Returns
        ---------
        s : complex array-like
            scattering parameters



        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/impedance_parameters
        .. [2] http://en.wikipedia.org/wiki/S-parameters
        .. [3] Kurokawa, Kaneyuki "Power waves and the scattering matrix", IEEE Transactions on Microwave Theory and Techniques, vol.13, iss.2, pp. 194–202, March 1965.
        .. [4] Marks, R. B. and Williams, D. F. "A general waveguide circuit theory", Journal of Research of National Institute of Standard and Technology, vol.97, iss.5, pp. 533–562, 1992.
        
        """
        nfreqs, nports, nports = z.shape
        z0 = self._fix_z0_shape(z0, nfreqs, nports)

        # Add a small real part in case of pure imaginary char impedance
        # to prevent numerical errors for both pseudo and power waves definitions
        z0 = z0.astype(dtype=npy.complex)
        z0[z0.real == 0] += ZERO    

        if s_def == 'power':
            # Power-waves. Eq.(18) from [3]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs 
            F, G = npy.zeros_like(z), npy.zeros_like(z)
            npy.einsum('ijj->ij', F)[...] = 1.0/npy.sqrt(z0.real)*0.5
            npy.einsum('ijj->ij', G)[...] = z0
            # s = F @ (z - npy.conjugate(G)) @ npy.linalg.inv(z + G) @ npy.linalg.inv(F)  # Python > 3.5
            s = npy.matmul(F, 
                        npy.matmul((z - npy.conjugate(G)), 
                                    npy.matmul(npy.linalg.inv(z + G), npy.linalg.inv(F))))


        elif s_def == 'pseudo':    
            # Pseudo-waves. Eq.(73) from [4]
            # Creating diagonal matrices of shape (nports,nports) for each nfreqs
            ZR, U = npy.zeros_like(z), npy.zeros_like(z)
            npy.einsum('ijj->ij', U)[...] = npy.sqrt(z0.real)/npy.abs(z0)
            npy.einsum('ijj->ij', ZR)[...] = z0
            # s = U @ (z - ZR) @ npy.linalg.inv(z + ZR) @ npy.linalg.inv(U)  # Python > 3.5
            s = npy.matmul(U, 
                        npy.matmul((z - ZR),
                                    npy.matmul(npy.linalg.inv(z + ZR), npy.linalg.inv(U))))

        elif s_def == 'traveling':
            # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
            # Creating Identity matrices of shape (nports,nports) for each nfreqs 
            Id = npy.zeros_like(z)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', Id)[...] = 1.0  
            # Creating diagonal matrices of shape (nports, nports) for each nfreqs
            sqrty0 = npy.zeros_like(z)  # (nfreqs, nports, nports)
            npy.einsum('ijj->ij', sqrty0)[...] = npy.sqrt(1.0/z0)
            # z -> s 
            s = npy.zeros_like(z)
            # s = (sqrty0 @ z @ sqrty0 - Id) @  npy.linalg.inv(sqrty0 @ z @ sqrty0 + Id)  # Python>3.5
            s = npy.matmul((npy.matmul(npy.matmul(sqrty0, z), sqrty0) - Id), 
                            npy.linalg.inv(npy.matmul(npy.matmul(sqrty0, z), sqrty0) + Id))
    
        
        return s

    def _fix_z0_shape(self, z0, nfreqs, nports):
        '''
        Make a port impedance of correct shape for a given network's matrix

        This attempts to broadcast z0 to satisfy
            npy.shape(z0) == (nfreqs,nports)

        Parameters
        --------------
        z0 : number, array-like
            z0 can be:
            * a number (same at all ports and frequencies)
            * an array-like of length == number ports.
            * an array-like of length == number frequency points.
            * the correct shape ==(nfreqs,nports)

        nfreqs : int
            number of frequency points
        nports : int
            number of ports

        Returns
        ----------
        z0 : array of shape ==(nfreqs,nports)
            z0  with the right shape for a nport Network

        Examples
        ----------
        For a two-port network with 201 frequency points, possible uses may
        be

        >>> z0 = rf.fix_z0_shape(50 , 201,2)
        >>> z0 = rf.fix_z0_shape([50,25] , 201,2)
        >>> z0 = rf.fix_z0_shape(range(201) , 201,2)


        '''

        if npy.shape(z0) == (nfreqs, nports):
            # z0 is of correct shape. super duper.return it quick.
            return z0.copy()

        elif npy.isscalar(z0):
            # z0 is a single number
            return npy.array(nfreqs * [nports * [z0]])

        elif len(z0) == nports:
            # assume z0 is a list of impedances for each port,
            # but constant with frequency
            return npy.array(nfreqs * [z0])

        elif len(z0) == nfreqs:
            # assume z0 is a list of impedances for each frequency,
            # but constant with respect to ports
            return npy.array(nports * [z0]).T

        else:
            raise IndexError('z0 is not an acceptable shape')

    def _t2s(self, t):
        '''
        converts scattering transfer parameters [#]_ to scattering parameters [#]_

        transfer parameters are also referred to as
        'wave cascading matrix', this function only operates on 2N-ports
        networks with same number of input and output ports, also known as
        'balanced networks'.

        Parameters
        -----------
        t : :class:`numpy.ndarray` (shape fx2nx2n)
                scattering transfer parameters

        Returns
        -------
        s : :class:`numpy.ndarray`
                scattering parameter matrix.

        See Also
        ---------
        inv : calculates inverse s-parameters
        s2z
        s2y
        s2t
        z2s
        z2y
        z2t
        y2s
        y2z
        y2z
        t2s
        t2z
        t2y
        Network.s
        Network.y
        Network.z
        Network.t

        References
        -----------
        .. [#] http://en.wikipedia.org/wiki/Scattering_transfer_parameters#Scattering_transfer_parameters
        .. [#] http://en.wikipedia.org/wiki/S-parameters
        .. [#] Janusz A. Dobrowolski, "Scattering Parameter in RF and Microwave Circuit Analysis and Design",
            Artech House, 2016, pp. 65-68
        '''
        z, y, x = t.shape
        # test here for even number of ports.
        # t-parameter networks are square matrix, so x and y are equal.
        if(x % 2 != 0):
            raise IndexError('Network don\'t have an even number of ports')
        s = npy.zeros((z, y, x), dtype=complex)
        yh = int(y/2)
        xh = int(x/2)
        # T_II,II^-1
        tinv = npy.linalg.inv(t[:, yh:y, xh:x])
        # np.linalg.inv test for singularity (matrix not invertible)
        for k in range(len(s)):
            # S_I,I = T_I,II . T_II,II^-1
            s[k, 0:yh, 0:xh] = t[k, 0:yh, xh:x].dot(tinv[k])
            # S_I,II = T_I,I - T_I,I,II . T_II,II^-1 . T_II,I
            s[k, 0:yh, xh:x] = t[k, 0:yh, 0:xh]-t[k, 0:yh, xh:x].dot(tinv[k].dot(t[k, yh:y, 0:xh]))
            # S_II,I = T_II,II^-1
            s[k, yh:y, 0:xh] = tinv[k]
            # S_II,II = -T_II,II^-1 . T_II,I
            s[k, yh:y, xh:x] = -tinv[k].dot(t[k, yh:y, 0:xh])
        return s




