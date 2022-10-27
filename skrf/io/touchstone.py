"""
.. module:: skrf.io.touchstone

========================================
touchstone (:mod:`skrf.io.touchstone`)
========================================

Touchstone class and utilities

.. autosummary::
   :toctree: generated/

   Touchstone

Functions related to reading/writing touchstones.
-------------------------------------------------

.. autosummary::
   :toctree: generated/

   hfss_touchstone_2_gamma_z0
   hfss_touchstone_2_media
   hfss_touchstone_2_network
   read_zipped_touchstones

"""
import re
import os
import typing
import zipfile
import numpy
import numpy as npy

from ..util import get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import Media, DefinedGammaZ0
from .. import mathFunctions as mf


class Touchstone:
    """
    Class to read touchstone s-parameter files.

    The reference for writing this class is the draft of the
    Touchstone(R) File Format Specification Rev 2.0 [#]_ and
    Touchstone(R) File Format Specification Version 2.0 [#]_

    References
    ----------
    .. [#] https://ibis.org/interconnect_wip/touchstone_spec2_draft.pdf
    .. [#] https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf
    """
    def __init__(self, file: typing.Union[str, typing.TextIO],
                 encoding: typing.Union[str, None] = None):
        """
        constructor

        Parameters
        ----------
        file : str or file-object
            touchstone file to load
        encoding : str, optional
            define the file encoding to use. Default value is None, 
            meaning the encoding is guessed (ANSI, UTF-8 or Latin-1).

        Examples
        --------
        From filename

        >>> t = rf.Touchstone('network.s2p')
        
        File encoding can be specified to help parsing the special characters:
        
        >>> t = rf.Touchstone('network.s2p', encoding='ISO-8859-1')

        From file-object

        >>> file = open('network.s2p')
        >>> t = rf.Touchstone(file)
        
        From a io.StringIO object
        
        >>> link = 'https://raw.githubusercontent.com/scikit-rf/scikit-rf/master/examples/basic_touchstone_plotting/horn antenna.s1p'
        >>> r = requests.get(link)
        >>> stringio = io.StringIO(r.text)
        >>> stringio.name = 'horn.s1p'  # must be provided for the Touchstone parser
        >>> horn = rf.Touchstone(stringio)

        """
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

        self.comment_variables = None

        # Does the input file have HFSS per frequency port impedances
        self.has_hfss_port_impedances = False

        # open the file depending on encoding
        # Guessing the encoding by trial-and-error, unless specified encoding
        try:
            try:
                if encoding is not None:
                    fid = get_fid(file, encoding=encoding)
                    self.filename = fid.name
                    self.load_file(fid)
                else:
                    # Assume default encoding
                    fid = get_fid(file)
                    self.filename = fid.name
                    self.load_file(fid)
            except Exception as e:
                fid.close()
                raise e

        except UnicodeDecodeError:
            # Unicode fails -> Force Latin-1
            fid = get_fid(file, encoding='ISO-8859-1')
            self.filename = fid.name
            self.load_file(fid)

        except ValueError:
            # Assume Microsoft UTF-8 variant encoding with BOM
            fid = get_fid(file, encoding='utf-8-sig')
            self.filename = fid.name
            self.load_file(fid)

        except Exception as e:
            raise ValueError(f'Something went wrong by the file opening: {e}')

        finally:
            self.gamma = []
            self.z0 = []

            if self.has_hfss_port_impedances:
                self.get_gamma_z0_from_fid(fid)

            fid.close()

    def load_file(self, fid: typing.TextIO):
        """
        Load the touchstone file into the internal data structures.

        Parameters
        ----------
        fid : file object

        """

        filename = self.filename

        # Check the filename extension.
        # Should be .sNp for Touchstone format V1.0, and .ts for V2
        extension = filename.split('.')[-1].lower()

        if (extension[0] == 's') and (extension[-1] == 'p'): # sNp
            # check if N is a correct number
            try:
                self.rank = int(extension[1:-1])
            except (ValueError):
                raise (ValueError("filename does not have a s-parameter extension. It has  [%s] instead. please, correct the extension to of form: 'sNp', where N is any integer." %(extension)))
        elif extension == 'ts':
            pass
        else:
            raise Exception('Filename does not have the expected Touchstone extension (.sNp or .ts)')

        values = []
        while True:
            line = fid.readline()
            if not line:
                break
            # store comments if they precede the option line
            line = line.split('!', 1)
            if len(line) == 2:
                if not self.parameter:
                    if self.comments is None:
                        self.comments = ''
                    self.comments = self.comments + line[1]
                elif line[1].startswith(' Port['):
                    try:
                        port_string, name = line[1].split('=', 1) #throws ValueError on unpack
                        name = name.strip()
                        garbage, index = port_string.strip().split('[', 1) #throws ValueError on unpack
                        index = int(index.rstrip(']')) #throws ValueError on not int-able
                        if index > self.rank or index <= 0:
                            print(f"Port name {name} provided for port number {index} but that's out of range for a file with extension s{self.rank}p")
                        else:
                            if self.port_names is None: #Initialize the array at the last minute
                                self.port_names = [''] * self.rank
                            self.port_names[index - 1] = name
                    except ValueError as e:
                        print(f"Error extracting port names from line: {line}")
                elif line[1].strip().lower().startswith('port impedance'):
                    self.has_hfss_port_impedances = True

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
                self.resistance = complex(toks[4])
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
            # we're separating them later
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

    def get_comments(self, ignored_comments=['Created with skrf']):
        """
        Returns the comments which appear anywhere in the file.

        Comment lines containing ignored comments are removed.
        By default these are comments which contain special meaning withing
        skrf and are not user comments.

        Returns
        -------
        processed_comments : string

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
        """
        Convert hfss variable comments to a dict of vars.

        Returns
        -------
        var_dict : dict (numbers, units)
            Dictionnary containing the comments
        """
        comments = self.comments
        p1 = re.compile(r'\w* = \w*.*')
        p2 = re.compile(r'\s*(\d*\.?\d*)\s*(\w*)')
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
        Returns the file format string used for the given format.

        This is useful to get some information.

        Returns
        -------
        format : string

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
        Generate a list of column names for the s-parameter data.
        The names are different for each format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db, orig (where orig refers to one of the three others)

        Returns
        -------
        names : list
            list of strings

        """
        names = ['frequency']
        if format == 'orig':
            format = self.format
        ext1, ext2 = {'ri':('R','I'),'ma':('M','A'), 'db':('DB','A')}.get(format)
        for r1 in range(self.rank):
            for r2 in range(self.rank):
                names.append("S%i%i%s"%(r1+1,r2+1,ext1))
                names.append("S%i%i%s"%(r1+1,r2+1,ext2))
        return names

    def get_sparameter_data(self, format='ri'):
        """
        Get the data of the s-parameter with the given format.

        Parameters
        ----------
        format : str
          Format: ri, ma, db, orig

        supported formats are:
          orig:  unmodified s-parameter data
          ri:    data in real/imaginary
          ma:    data in magnitude and angle (degree)
          db:    data in log magnitude and angle (degree)

        Returns
        -------
        ret: list
            list of numpy.arrays

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
                v_complex = values[:,1::2] + 1j* values[:,2::2]
                values[:,1::2] = numpy.absolute(v_complex)
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)
            elif (self.format == 'ri') and (format == 'db'):
                v_complex = values[:,1::2] + 1j* values[:,2::2]
                values[:,1::2] = 20*numpy.log10(numpy.absolute(v_complex))
                values[:,2::2] = numpy.angle(v_complex)*(180/numpy.pi)

        for i,n in enumerate(self.get_sparameter_names(format=format)):
            ret[n] = values[:,i]

        # transpose Touchstone V1 2-port files (.2p), as the order is (11) (21) (12) (22)
        file_name_ending = self.filename.split('.')[-1].lower()
        if self.rank == 2 and file_name_ending == "s2p":
            swaps = [ k for k in ret if '21' in k]
            for s in swaps:
                true_s = s.replace('21', '12')
                ret[s], ret[true_s] = ret[true_s], ret[s]

        return ret

    def get_sparameter_arrays(self):
        """
        Returns the s-parameters as a tuple of arrays.

        The first element is the frequency vector (in Hz) and the s-parameters are a 3d numpy array.
        The values of the s-parameters are complex number.

        Returns
        -------
        param : tuple of arrays

        Examples
        --------
        >>> f, a = self.sgetparameter_arrays()
        >>> s11 = a[:, 0, 0]

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
        raise NotImplementedError('not yet implemented')


    def get_noise_data(self):
        # TBD = 1
        # noise_frequency = noise_values[:,0]
        # noise_minimum_figure = noise_values[:,1]
        # noise_source_reflection = noise_values[:,2]
        # noise_source_phase = noise_values[:,3]
        # noise_normalized_resistance = noise_values[:,4]
        raise NotImplementedError('not yet implemented')

    def get_gamma_z0_from_fid(self, fid):
        """
        Extracts Z0 and Gamma comments from fid.

        Parameters
        ----------
        fid : file object
        """
        gamma = []
        z0 = []
        def line2ComplexVector(s):
            return mf.scalar2Complex(npy.array([k for k in s.strip().split(' ')
                                                if k != ''][self.rank*-2:],
                                                dtype='float'))
        fid.seek(0)
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.replace('\t', ' ')

            # HFSS adds gamma and z0 data in .sNp files using comments.
            # NB : each line(s) describe gamma and z0.
            #  But, depending on the HFSS version, either:
            #  - up to 4 ports only.
                #  - up to 4 ports only.
            #        for N > 4, gamma and z0 are given by additional lines
            #  - all gamma and z0 are given on a single line (since 2020R2)
            # In addition, some spurious '!' can remain in these lines
            if '! Gamma' in line:
                _line = line.replace('! Gamma', '').replace('!', '').rstrip()

                # check how many elements are in the first line
                nb_elem = len(_line.split())

                if nb_elem == 2*self.rank:
                    # case of all data in a single line
                    gamma.append(line2ComplexVector(_line.replace('!', '').rstrip()))
                else:
                    # case of Nport > 4 *and* data on additional multiple lines
                    for _ in range(int(npy.ceil(self.rank/4.0)) - 1):
                        _line += fid.readline().replace('!', '').rstrip()
                    gamma.append(line2ComplexVector(_line))


            if '! Port Impedance' in line:
                _line = line.replace('! Port Impedance', '').rstrip()
                nb_elem = len(_line.split())

                if nb_elem == 2*self.rank:
                    z0.append(line2ComplexVector(_line.replace('!', '').rstrip()))
                else:
                    for _ in range(int(npy.ceil(self.rank/4.0)) - 1):
                        _line += fid.readline().replace('!', '').rstrip()
                    z0.append(line2ComplexVector(_line))

        # If the file does not contain valid port impedance comments, set to default one
        if len(z0) == 0:
            z0 = npy.array(self.resistance, dtype=complex)
            #raise ValueError('Touchstone does not contain valid gamma, port impedance comments')

        self.gamma = npy.array(gamma)
        self.z0 = npy.array(z0)

    def get_gamma_z0(self):
        """
        Extracts Z0 and Gamma comments from touchstone file (if provided).

        Returns
        -------
        gamma : complex npy.ndarray
            complex  propagation constant
        z0 : npy.ndarray
            complex port impedance
        """
        return self.gamma, self.z0

def hfss_touchstone_2_gamma_z0(filename):
    """
    Extracts Z0 and Gamma comments from touchstone file.

    Takes a HFSS-style Touchstone file with Gamma and Z0 comments and
    extracts a triplet of arrays being: (frequency, Gamma, Z0)

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file


    Returns
    -------
    f : npy.ndarray
        frequency vector (in Hz)
    gamma : complex npy.ndarray
        complex  propagation constant
    z0 : npy.ndarray
        complex port impedance

    Examples
    --------
    >>> f,gamm,z0 = rf.hfss_touchstone_2_gamma_z0('line.s2p')
    """
    ntwk = Network(filename)

    return ntwk.frequency.f, ntwk.gamma, ntwk.z0


def hfss_touchstone_2_media(filename, f_unit='ghz'):
    """
    Creates a :class:`~skrf.media.Media` object from a a HFSS-style Touchstone file with Gamma and Z0 comments.

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file
    f_unit : string
        'hz', 'khz', 'mhz' or 'ghz', which is passed to the `f_unit` parameter
        to :class:`~skrf.frequency.Frequency` constructor

    Returns
    -------
    my_media : :class:`~skrf.media.media.Media` object
        the transmission line model defined by the gamma, and z0
        comments in the HFSS file.

    Examples
    --------
    >>> port1_media, port2_media = rf.hfss_touchstone_2_media('line.s2p')

    See Also
    --------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0
    """
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
    """
    Creates a :class:`~skrf.Network` object from a a HFSS-style Touchstone file.

    Parameters
    ----------
    filename : string
        the HFSS-style Touchstone file
    f_unit : string
        'hz', 'khz', 'mhz' or 'ghz', which is passed to the `f_unit` parameter
        to :class:`~skrf.frequency.Frequency` constructor

    Returns
    -------
    my_network : :class:`~skrf.network.Network` object
        the n-port network model

    Examples
    --------
    >>> my_network = rf.hfss_touchstone_2_network('DUT.s2p')

    See Also
    --------
    hfss_touchstone_2_gamma_z0 : returns gamma, and z0
    """
    my_network = Network(file=filename, f_unit=f_unit)
    return(my_network)


def read_zipped_touchstones(ziparchive: zipfile.ZipFile, dir: str = "") -> typing.Dict[str, Network]:
    """
    similar to skrf.io.read_all_networks, which works for directories but only for Touchstones in ziparchives.

    Parameters
    ----------
    ziparchive : :class:`zipfile.ZipFile`
        an zip archive file, containing Touchstone files and open for reading
    dir : str
        the directory of the ziparchive to read networks from, default is "" which reads only the root directory

    Returns
    -------
    dict
        keys are touchstone filenames without extensions
        values are network objects created from the touchstone files
    """
    networks = dict()
    for fname in ziparchive.namelist():  # type: str
        directory, filename = os.path.split(fname)
        if dir == directory and fname[-4:].lower() in (".s1p", ".s2p", ".s3p", ".s4p"):
            network = Network.zipped_touchstone(fname, ziparchive)
            networks[network.name] = network
    return networks
