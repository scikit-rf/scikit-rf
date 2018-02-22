"""
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
    hfss_touchstone_2_network
"""
import re
import os
import zipfile
import numpy
import numpy as npy

from six.moves import xrange

from ..util import get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import Media, DefinedGammaZ0
from .. import mathFunctions as mf


class Touchstone:
    """
    class to read touchstone s-parameter files

    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0 [#]_ and
    Touchstone(R) File Format Specification Version 2.0 [##]_

    .. [#] https://ibis.org/interconnect_wip/touchstone_spec2_draft.pdf
    .. [##] https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf
    """
    def __init__(self, file):
        """
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
        """
        fid = get_fid(file)
        filename = fid.name
        ## file name of the touchstone data file
        self.filename = filename

        ## file format version. 
        # Defined by default to 1.0, since version number can be omitted in V1.0 format
        self.version = '1.0'
        ## comments in the file header
        self.comments = None
        ## unit of the frequency (Hz, kHz, MHz, GHz)
        self.frequency_unit = None
        ## number of frequency points 
        self.frequency_nb = None
        ## s-parameter type (S,Y,Z,G,H)
        self.parameter = None
        ## s-parameter format (MA, DB, RI)
        self.format = None
        ## reference resistance, global setup
        self.resistance = None
        ## reference impedance for each s-parameter
        self.reference = None

        ## numpy array of original s-parameter data
        self.sparameters = None
        ## numpy array of original noise data
        self.noise = None

        ## kind of s-parameter data (s1p, s2p, s3p, s4p)
        self.rank = None
        ## Store port names in a list if they exist in the file
        self.port_names = None

        self.comment_variables=None
        self.load_file(fid)

    def load_file(self, fid):
        """
        Load the touchstone file into the interal data structures
        """

        filename=self.filename

        # Check the filename extension. 
        # Should be .sNp for Touchstone format V1.0, and .ts for V2
        extension = filename.split('.')[-1].lower()
        
        if (extension[0] == 's') and (extension[-1] == 'p'): # sNp
            # check if N is a correct unmber
            try:
                self.rank = int(extension[1:-1])
            except (ValueError):
                raise (ValueError("filename does not have a s-parameter extension. It has  [%s] instead. please, correct the extension to of form: 'sNp', where N is any integer." %(extention)))
        elif extension == 'ts':
            pass
        else:
            raise Exception('Filename does not have the expected Touchstone extension (.sNp or .ts)')

        linenr = 0
        values = []
        while (1):
            linenr +=1
            line = fid.readline()
            if not type(line) == str:
                line = line.decode("ascii")  # for python3 zipfile compatibility
            if not line:
                break

            # store comments if they precede the option line
            line = line.split('!',1)
            if len(line) == 2:
                if not self.parameter:
                    if self.comments == None:
                        self.comments = ''
                    self.comments = self.comments + line[1]
                elif line[1].startswith(' Port['):
                    try:
                        port_string, name = line[1].split('=', 1) #throws ValueError on unpack
                        name = name.strip()
                        garbage, index = port_string.strip().split('[', 1) #throws ValueError on unpack
                        index = int(index.rstrip(']')) #throws ValueError on not int-able
                        if index > self.rank or index <= 0:
                            print("Port name {0} provided for port number {1} but that's out of range for a file with extension s{2}p".format(name, index, self.rank))
                        else:
                            if self.port_names is None: #Initialize the array at the last minute
                                self.port_names = [''] * self.rank
                            self.port_names[index - 1] = name
                    except ValueError as e:
                        print("Error extracting port names from line: {0}".format(line))

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
                # The reference impedances can be span after the keyword  
                # or on the following line
                self.reference = [ float(r) for r in line.split()[2:] ]
                if not self.reference:
                    line = fid.readline()
                    self.reference = [ float(r) for r in line.split()]
                continue
            
            # grab the [Number of Ports] string
            if line[:17] == '[number of ports]':
                self.rank = int(line.split()[-1])
                continue
            
            # grab the [Number of Frequencies] string
            if line[:23] == '[number of frequencies]':
                self.frequency_nb = line.split()[-1]
                continue

            # skip the [Network Data] keyword
            if line[:14] == '[network data]':
                continue
            
            # skip the [End] keyword
            if line[:5] == '[end]':
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
                    print('ERROR: illegal frequency_unit [%s]',  self.frequency_unit)
                    # TODO: Raise
                if self.parameter not in 'syzgh':
                    print('ERROR: illegal parameter value [%s]', self.parameter)
                    # TODO: Raise
                if self.format not in ['ma', 'db', 'ri']:
                    print('ERROR: illegal format value [%s]', self.format)
                    # TODO: Raise

                continue

            # collect all values without taking care of there meaning
            # we're seperating them later
            values.extend([ float(v) for v in line.split() ])

        # let's do some post-processing to the read values
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
    
    def get_comment_variables(self):
        '''
        convert hfss variable comments to a dict of vars:(numbers,units)
        '''
        comments = self.comments
        p1 = re.compile('\w* = \w*')
        p2 = re.compile('\s*(\d*)\s*(\w*)')
        var_dict = {}
        for k in re.findall(p1, comments):
            var, value = k.split('=')
            var=var.rstrip()
            try:
                var_dict[var] = p2.match(value).groups()
            except:
                pass
        return var_dict
    
    def get_format(self, format="ri"):
        """
        returns the file format string used for the given format.
        This is useful to get some information.
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
        get the data of the s-parameter with the given format.
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
        Returns the s-parameters as a tuple of arrays, where the first element is
        the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the s-parameters are complex number.
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
        
    def is_from_hfss(self):
        '''
        Check if the Touchstone file has been produced by HFSS
        
        Returns
        ------------
        status : boolean
            True if the Touchstone file has been produced by HFSS
            False otherwise
        '''    
        status = False
        if 'Exported from HFSS' in self.comments:
            status = True
        return status      
    
    def get_gamma_z0(self):
        '''
        Extracts Z0 and Gamma comments from touchstone file (is provided)
        
        Returns
        --------
        gamma : complex numpy.ndarray
            complex  propagation constant
        z0 : numpy.ndarray
            complex port impedance    
        '''
        def line2ComplexVector(s):
            return mf.scalar2Complex(npy.array([k for k in s.strip().split(' ')
                                                if k != ''][self.rank*-2:],
                                                dtype='float'))
    
        with open(self.filename) as f:
            gamma, z0 = [],[]
    
            for line in f:
                if '! Gamma' in line:
                    gamma.append(line2ComplexVector(line.replace('! Gamma', '')))
                if '! Port Impedance' in line:
                    z0.append(line2ComplexVector(line.replace('! Port Impedance', '')))
    
            # If the file does not contain valid port impedance comments, set to default one
            if len(z0) == 0:
                z0 = self.resistance
                #raise ValueError('Touchstone does not contain valid gamma, port impedance comments')


        return npy.array(gamma), npy.array(z0)

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
    ntwk = Network(filename)

    return ntwk.frequency.f, ntwk.gamma, ntwk.z0


def hfss_touchstone_2_media(filename, f_unit='ghz'):
    '''
    Creates a :class:`~skrf.media.Media` object from a a HFSS-style touchstone file with Gamma and Z0 comments

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
    ntwk = Network(filename)

    freq = ntwk.frequency
    gamma = ntwk.gamma
    z0 = ntwk.z0
    

    media_list = []

    for port_n in range(gamma.shape[1]):
        media_list.append(\
            DefinedGammaZ0(
                frequency = freq,
                gamma =  gamma[:, port_n],
                z0 = z0[:, port_n]
                )
            )


    return media_list


def hfss_touchstone_2_network(filename, f_unit='ghz'):
    '''
    Creates a :class:`~skrf.Network` object from a a HFSS-style touchstone file

    Parameters
    ------------
    filename : string
        the HFSS-style touchstone file
    f_unit : ['hz','khz','mhz','ghz']
        passed to f_unit parameters of Frequency constructor

    Returns
    --------
    my_network : skrf.Network object
        the n-port network model

    Examples
    ----------
    >>> my_network = rf.hfss_touchstone_2_network('DUT.s2p')

    See Also
    ---------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0
    '''
    my_network = Network(file=filename, f_unit=f_unit)
    return(my_network)


def read_zipped_touchstones(ziparchive, dir=""):
    """
    similar to skrf.io.read_all_networks, which works for directories but only for touchstones in ziparchives

    Parameters
    ----------
    ziparchive : zipfile.ZipFile
        an zip archive file, containing touchstone files and open for reading
    dir : str
        the directory of the ziparchive to read networks from, default is "" which reads only the root directory

    Returns
    -------
    dict
    """
    networks = dict()
    for fname in ziparchive.namelist():  # type: str
        directory, filename = os.path.split(fname)
        if dir == directory and fname[-4:].lower() in (".s1p", ".s2p", ".s3p", ".s4p"):
            network = Network.zipped_touchstone(fname, ziparchive)
            networks[network.name] = network
    return networks

