

'''
.. module:: skrf.io.touchstone
========================================
touchstone (:mod:`skrf.io.touchstone`)
========================================

Touchstone class

.. autosummary::
    :toctree: generated/

    Touchstone
    

Functions related to reading/writing touchstones.

.. autosummary::
    :toctree: generated/

    hfss_touchstone_2_gamma_z0
    hfss_touchstone_2_media
'''

import numpy
import numpy as npy
from ..util import get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import  Media
from .. import mathFunctions as mf

class Touchstone():
    '''
    class to read touchstone s-parameter files
    
    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0 [#]_
    
    .. [#] http://www.eda-stds.org/ibis/adhoc/interconnect/touchstone_spec2_draft.pdf
    '''
    def __init__(self,file):
        '''
        constructor
        
        Parameters
        -------------
        file : str or file-object
            touchstone file to load
            
        Examples
        ---------
        From filename 
        
        >>> t = rf.Touchstone('network.s2p')
        
        From file-object
        
        >>> file = open('network.s2p')
        >>> t = rf.Touchstone(file)
        '''
        fid = get_fid(file)
        filename = fid.name
        ## file name of the touchstone data file
        self.filename = filename

        ## file format version
        self.version = '1.0'
        ## comments in the file header
        self.comments = None
        ## unit of the frequency (Hz, kHz, MHz, GHz)
        self.frequency_unit = None
        ## s-parameter type (S,Y,Z,G,H)
        self.parameter = None
        ## s-parameter format (MA, DB, RI)
        self.format = None
        ## reference resistance, global setup
        self.resistance = None
        ## reference impedance for each s-parameter
        self.reference = None

        ## numpy array of original sparameter data
        self.sparameters = None
        ## numpy array of original noise data
        self.noise = None

        ## kind of s-parameter data (s1p, s2p, s3p, s4p)
        self.rank = None

        self.load_file(fid)

    def load_file(self, fid):
        """
        Load the touchstone file into the interal data structures
        """
        
        filename=self.filename
        
        extention = filename.split('.')[-1].lower()
        #self.rank = {'s1p':1, 's2p':2, 's3p':3, 's4p':4}.get(extention, None)
        try:
            self.rank = int(extention[1:-1])
        except (ValueError):
            raise (ValueError("filename does not have a s-parameter extention. It has  [%s] instead. please, correct the extension to of form: 'sNp', where N is any integer." %(extention)))


        linenr = 0
        values = []
        while (1):
            linenr +=1
            line = fid.readline()
            if not line:
                break

            # store comments if they precede the option line
            line = line.split('!',1)
            if len(line) == 2 and not self.parameter:
                if self.comments == None:
                    self.comments = ''
                self.comments = self.comments + line[1]
            
            # remove the comment (if any) so rest of line can be processed.
            # touchstone files are case-insensitive
            line = line[0].strip().lower()
            
            # skip the line if there was nothing except comments
            if len(line) == 0:
                continue

            # grab the [version] string
            if line[:9] == '[version]':
                self.version = line.split()[1]
                continue

            # grab the [reference] string
            if line[:11] == '[reference]':
                self.reference = [ float(r) for r in line.split()[2:] ]
                continue

            # the option line
            if line[0] == '#':
                toks = line[1:].strip().split()
                # fill the option line with the missing defaults
                toks.extend(['ghz', 's', 'ma', 'r', '50'][len(toks):])
                self.frequency_unit = toks[0]
                self.parameter = toks[1]
                self.format = toks[2]
                self.resistance = toks[4]
                if self.frequency_unit not in ['hz', 'khz', 'mhz', 'ghz']:
                    print 'ERROR: illegal frequency_unit [%s]',  self.frequency_unit
                    # TODO: Raise
                if self.parameter not in 'syzgh':
                    print 'ERROR: illegal parameter value [%s]', self.parameter
                    # TODO: Raise
                if self.format not in ['ma', 'db', 'ri']:
                    print 'ERROR: illegal format value [%s]', self.format
                    # TODO: Raise

                continue

            # collect all values without taking care of there meaning
            # we're seperating them later
            values.extend([ float(v) for v in line.split() ])

        # let's do some postprocessing to the read values
        # for s2p parameters there may be noise parameters in the value list
        values = numpy.asarray(values)
        if self.rank == 2:
            # the first frequency value that is smaller than the last one is the
            # indicator for the start of the noise section
            # each set of the s-parameter section is 9 values long
            pos = numpy.where(numpy.sign(numpy.diff(values[::9])) == -1)
            if len(pos[0]) != 0:
                # we have noise data in the values
                pos = pos[0][0] + 1   # add 1 because diff reduced it by 1
                noise_values = values[pos*9:]
                values = values[:pos*9]
                self.noise = noise_values.reshape((-1,5))

        if len(values)%(1+2*(self.rank)**2) != 0 :
            # incomplete data line / matrix found            
            raise AssertionError

        # reshape the values to match the rank
        self.sparameters = values.reshape((-1, 1 + 2*self.rank**2))
        # multiplier from the frequency unit
        self.frequency_mult = {'hz':1.0, 'khz':1e3,
                               'mhz':1e6, 'ghz':1e9}.get(self.frequency_unit)
        # set the reference to the resistance value if no [reference] is provided
        if not self.reference:
            self.reference = [self.resistance] * self.rank

    def get_comments(self, ignored_comments = ['Created with skrf']):
        """
        Returns the comments which appear anywhere in the file.  Comment lines
        containing ignored comments are removed.  By default these are comments
        which contain special meaning withing skrf and are not user comments.
        """
        processed_comments = '' 
        if self.comments is None:
            self.comments = ''    
        for comment_line in self.comments.split('\n'):
            for ignored_comment in ignored_comments:
                if ignored_comment in comment_line:
                        comment_line = None
            if comment_line:
                processed_comments = processed_comments + comment_line + '\n'
        return processed_comments                
        
    def get_format(self, format="ri"):
        """
        returns the file format string used for the given format.
        This is usefull to get some informations.
        """
        if format == 'orig':
            frequency = self.frequency_unit
            format = self.format
        else:
            frequency = 'hz'
        return "%s %s %s r %s" %(frequency, self.parameter,
                                 format, self.resistance)
                                 

    def get_sparameter_names(self, format="ri"):
        """
        generate a list of column names for the s-parameter data
        The names are different for each format.
        posible format parameters:
          ri, ma, db, orig  (where orig refers to one of the three others)
        returns a list of strings.
        """
        names = ['frequency']
        if format == 'orig':
            format = self.format
        ext1, ext2 = {'ri':('R','I'),'ma':('M','A'), 'db':('DB','A')}.get(format)
        for r1 in xrange(self.rank):
            for r2 in xrange(self.rank):
                names.append("S%i%i%s"%(r1+1,r2+1,ext1))
                names.append("S%i%i%s"%(r1+1,r2+1,ext2))
        return names

    def get_sparameter_data(self, format='ri'):
        """
        get the data of the sparameter with the given format.
        supported formats are:
          orig:  unmodified s-parameter data
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitute and angle (degree)
        Returns a list of numpy.arrays
        """
        ret = {}
        if format == 'orig':
            values = self.sparameters
        else:
            values = self.sparameters.copy()
            # use frequency in hz unit
            values[:,0] = values[:,0]*self.frequency_mult
            if (self.format == 'db') and (format == 'ma'):
                values[:,1::2] = 10**(values[:,1::2]/20.0)
            elif (self.format == 'db') and (format == 'ri'):
                v_complex = ((10**values[:,1::2]/20.0)
                             * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ma') and (format == 'db'):
                values[:,1::2] = 20*numpy.log10(values[:,1::2])
            elif (self.format == 'ma') and (format == 'ri'):
                v_complex = (values[:,1::2] * numpy.exp(1j*numpy.pi/180 * values[:,2::2]))
                values[:,1::2] = numpy.real(v_complex)
                values[:,2::2] = numpy.imag(v_complex)
            elif (self.format == 'ri') and (format == 'ma'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = numpy.absolute(v_complex)
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)
            elif (self.format == 'ri') and (format == 'db'):
                v_complex = numpy.absolute(values[:,1::2] + 1j* self.sparameters[:,2::2])
                values[:,1::2] = 20*numpy.log10(numpy.absolute(v_complex))
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)

        for i,n in enumerate(self.get_sparameter_names(format=format)):
            ret[n] = values[:,i]
        return ret

    def get_sparameter_arrays(self):
        """
        returns the sparameters as a tuple of arrays, where the first element is
        the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the sparameters are complex number.
        usage:
          f,a = self.sgetparameter_arrays()
          s11 = a[:,0,0]
        """
        v = self.sparameters

        if self.format == 'ri':
            v_complex = v[:,1::2] + 1j* v[:,2::2]
        elif self.format == 'ma':
            v_complex = (v[:,1::2] * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))
        elif self.format == 'db':
            v_complex = ((10**(v[:,1::2]/20.0)) * numpy.exp(1j*numpy.pi/180 * v[:,2::2]))

        if self.rank == 2 :
            # this return is tricky; it handles the way touchtone lines are 
            # in case of rank==2: order is s11,s21,s12,s22 
            return (v[:,0] * self.frequency_mult,
                    numpy.transpose(v_complex.reshape((-1, self.rank, self.rank)),axes=(0,2,1)))
        else:
            return (v[:,0] * self.frequency_mult,
                    v_complex.reshape((-1, self.rank, self.rank)))

    def get_noise_names(self):
        """
        TODO: NIY
        """
        TBD = 1


    def get_noise_data(self):
        """
        TODO: NIY
        """
        TBD = 1
        noise_frequency = noise_values[:,0]
        noise_minimum_figure = noise_values[:,1]
        noise_source_reflection = noise_values[:,2]
        noise_source_phase = noise_values[:,3]
        noise_normalized_resistance = noise_values[:,4]


def hfss_touchstone_2_gamma_z0(filename):
    '''
    Extracts Z0 and Gamma comments from touchstone file
    
    Takes a HFSS-style touchstone file with Gamma and Z0 comments and 
    extracts a triplet of arrays being: (frequency, Gamma, Z0)
    
    Parameters
    ------------
    filename : string 
        the HFSS-style touchstone file
    
    
    Returns
    --------
    f : numpy.ndarray
        frequency vector (in Hz)
    gamma : complex numpy.ndarray
        complex  propagation constant
    z0 : numpy.ndarray
        complex port impedance
    
    Examples
    ----------
    >>> f,gamm,z0 = rf.hfss_touchstone_2_gamma_z0('line.s2p')
    '''
    #TODO: make this work for different HFSS versions. and arbitrary 
    # number of ports
    ntwk = Network(filename)
    f= open(filename)
    gamma, z0 = [],[]
    
    def line2ComplexVector(s):
        return mf.scalar2Complex(\
            npy.array(\
                [k for k in s.strip().split(' ') if k != ''][ntwk.nports*-2:],\
                dtype='float'
                )
            )
            
    for line in f:
        if '! Gamma' in line:
            gamma.append(line2ComplexVector(line))
        if '! Port Impedance' in line:
            z0.append(line2ComplexVector(line))
    
    if len (z0) ==0:
        raise(ValueError('Touchstone does not contain valid gamma, port impedance comments'))
        
    return ntwk.frequency.f, npy.array(gamma), npy.array(z0)

def hfss_touchstone_2_media(filename, f_unit='ghz'):
    '''
    Creates a :class:`~skrf.media.media.Media` object from a a HFSS-style touchstone file with Gamma and Z0 comments 
    
    Parameters
    ------------
    filename : string 
        the HFSS-style touchstone file
    f_unit : ['hz','khz','mhz','ghz']
        passed to f_unit parameters of Frequency constructor
    
    Returns
    --------
    my_media : skrf.media.Media object
        the transmission line model defined by the gamma, and z0 
        comments in the HFSS file.
    
    Examples
    ----------
    >>> port1_media, port2_media = rf.hfss_touchstone_2_media('line.s2p')
    
    See Also
    ---------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0 
    '''
    f, gamma, z0 = hfss_touchstone_2_gamma_z0(filename)
    
    freq = Frequency.from_f(f)
    freq.unit = f_unit
    
    
    media_list = []
    
    for port_n in range(gamma.shape[1]):
        media_list.append(\
            Media(
                frequency = freq, 
                propagation_constant =  gamma[:, port_n],
                characteristic_impedance = z0[:, port_n]
                )
            )
        
        
    return media_list 
