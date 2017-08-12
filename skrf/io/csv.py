
'''
.. module:: skrf.io.csv
========================================
csv (:mod:`skrf.io.csv`)
========================================

Functions for reading and writing standard csv files

.. autosummary::
    :toctree: generated/

    read_pna_csv
    pna_csv_2_ntwks
'''
import numpy as npy
import os
from ..network import Network
from .. import mathFunctions as mf
from ..frequency import Frequency
from .. import util
from warnings import warn

# delayed imports
# from pandas import Series, Index, DataFrame

def read_pna_csv(filename, *args, **kwargs):
    '''
    Reads data from a csv file written by an Agilient PNA.

    This function returns a triplet containing the header, comments,
    and data.


    Parameters
    -------------
    filename : str
        the file
    \*args, \*\*kwargs :

    Returns
    ---------
    header : str
        The header string, which is the line following the 'BEGIN'
    comments : str
        All lines that begin with a '!'
    data : :class:`numpy.ndarray`
        An array containing the data. The meaning of which depends on
        the header.

    See Also
    ----------
    pna_csv_2_ntwks : Reads a csv file which contains s-parameter data

    Examples
    -----------
    >>> header, comments, data = rf.read_pna_csv('myfile.csv')
    '''
    warn("deprecated", DeprecationWarning)
    with open(filename,'r') as fid:
        begin_line = -2
        end_line = -1
        n_END = 0
        comments = ''
        for k,line in enumerate(fid.readlines()):
            if line.startswith('!'):
                comments += line[1:]
            elif line.startswith('BEGIN') and n_END == 0:
                begin_line = k
            elif line.startswith('END'):
                if n_END == 0:
                #first END spotted -> set end_line to read first data block only
                    end_line = k
                #increment n_END to allow for CR correction in genfromtxt
                n_END += 1
            if k == begin_line+1:
                header = line
        footer = k - end_line

    try:
        data = npy.genfromtxt(
            filename,
            delimiter = ',',
            skip_header = begin_line + 2,
            skip_footer = footer - (n_END-1)*2,
            *args, **kwargs
            )
    except(ValueError):
        # carrage returns require a doubling of skiplines
        data = npy.genfromtxt(
            filename,
            delimiter = ',',
            skip_header = (begin_line + 2)*2,
            skip_footer = footer,
            *args, **kwargs
            )

    # pna uses unicode coding for degree symbol, but we dont need that
    header = header.replace('\xb0','deg').rstrip('\n').rstrip('\r')

    return header, comments, data

def pna_csv_2_df(filename, *args, **kwargs):
    '''
    Reads data from a csv file written by an Agilient PNA

    '''
    warn("deprecated", DeprecationWarning)
    from pandas import Series, Index, DataFrame
    header, comments, d = read_pna_csv(filename)
    f_unit = header.split(',')[0].split(')')[0].split('(')[1]

    names = header.split(',')

    index = Index(d[:,0], name = names[0])
    df=DataFrame(dict([(names[k], d[:,k]) for k in range(1,len(names))]), index=index)
    return df

def pna_csv_2_ntwks2(filename, *args, **kwargs):
    warn("deprecated", DeprecationWarning)
    df = pna_csv_2_df(filename, *args, **kwargs)
    header, comments, d = read_pna_csv(filename)
    ntwk_dict  = {}
    param_set=set([k[:3] for k in df.columns])
    f = df.index.values*1e-9
    for param in param_set:
        try:
            s = mf.dbdeg_2_reim(
                df['%s Log Mag(dB)'%param].values,
                df['%s Phase(deg)'%param].values,
                )
        except(KeyError):
            s = mf.dbdeg_2_reim(
                df['%s (REAL)'%param].values,
                df['%s (IMAG)'%param].values,
                )

        ntwk_dict[param] = Network(f=f, s=s, name=param, comments=comments)


    try:
        s=npy.zeros((len(f),2,2), dtype=complex)
        s[:,0,0] = ntwk_dict['S11'].s.flatten()
        s[:,1,1] = ntwk_dict['S22'].s.flatten()
        s[:,1,0] = ntwk_dict['S21'].s.flatten()
        s[:,0,1] = ntwk_dict['S12'].s.flatten()
        name  =os.path.splitext(os.path.basename(filename))[0]
        ntwk = Network(f=f, s=s, name=name, comments=comments)

        return ntwk
    except:
        return ntwk_dict

def pna_csv_2_ntwks3(filename):
    '''
    Read a CSV file exported from an Agilent PNA in dB/deg format

    Parameters
    --------------
    filename : str
        full path or filename

    Returns
    ---------
    out : n
        2-Port Network

    Examples
    ----------

    '''
    header, comments, d = read_pna_csv(filename)
    col_headers = pna_csv_header_split(filename)

    # set impedance to 50 Ohm (doesn't matter for now)
    z0 = npy.ones((npy.shape(d)[0]))*50
    # read f values, convert to GHz
    f = d[:,0]/1e9

    name = os.path.splitext(os.path.basename(filename))[0]

    if 'db' in header.lower() and 'deg' in header.lower():
        # this is a cvs in DB/DEG format
        # -> convert db/deg values to real/imag values
        s = npy.zeros((len(f),2,2), dtype=complex)

        for k, h in enumerate(col_headers[1:]):
            if 's11' in h.lower() and 'db' in h.lower():
                s[:,0,0] = mf.dbdeg_2_reim(d[:,k+1], d[:,k+2])
            elif 's21' in h.lower() and 'db' in h.lower():
                s[:,1,0] = mf.dbdeg_2_reim(d[:,k+1], d[:,k+2])
            elif 's12' in h.lower() and 'db' in h.lower():
                s[:,0,1] = mf.dbdeg_2_reim(d[:,k+1], d[:,k+2])
            elif 's22' in h.lower() and 'db' in h.lower():
                s[:,1,1] = mf.dbdeg_2_reim(d[:,k+1], d[:,k+2])

        n = Network(f=f,s=s,z0=z0, name = name)
        return n

    else:
        warn("File does not seem to be formatted properly (only dB/deg supported for now)")

def read_all_csv(dir='.', contains = None):
    '''
    Read all CSV files in a directory

    Parameters
    --------------
    dir : str, optional
        the directory to load from, default  \'.\'
    contains : str, optional
        if not None, only files containing this substring will be loaded

    Returns
    ---------
    out : dictionary
        dictionary containing all loaded CSV objects. keys are the
        filenames without extensions, and the values are the objects


    Examples
    ----------


    See Also
    ----------

    '''

    out={}
    for filename in os.listdir(dir):
        if contains is not None and contains not in filename:
            continue
        fullname = os.path.join(dir,filename)
        keyname = os.path.splitext(filename)[0]
        try:
            out[keyname] = pna_csv_2_ntwks3(fullname)
            continue
        except:
            pass

        try:
            out[keyname] = Network(fullname)
            continue
        except:
            pass

    return out


class AgilentCSV(object):
    '''
    Agilent-style csv file representing either scalar traces vs frequency
    or complex data vs. frequency


    '''
    def __init__(self, filename, *args, **kwargs):
        '''
        Init.

        Parameters
        ----------
        filename : str
            filename
        \*args ,\*\*kwargs :
            passed to Network.__init__ in :func:`networks` and :func:`scalar_networks`
        '''
        self.filename = filename
        self.header, self.comments, self.data = self.read()
        self.args, self.kwargs = args, kwargs

    def read(self):
        '''
        Reads data from  file

        This function returns a triplet containing the header, comments,
        and data.

        Returns
        ---------
        header : str
            The header string, which is the line following the 'BEGIN'
        comments : str
            All lines that begin with a '!'
        data : :class:`numpy.ndarray`
            An array containing the data. The meaning of which depends on
            the header.
        '''
        with open(self.filename, 'r') as fid:
            begin_line = -2
            end_line = -1
            comments = ''
            for k,line in enumerate(fid.readlines()):
                if line.startswith('!'):
                    comments += line[1:]
                elif line.startswith('BEGIN'):
                    begin_line = k
                elif line.startswith('END'):
                    end_line = k

                if k == begin_line+1:
                    header = line

            footer = k - end_line

        try:
            data = npy.genfromtxt(
                self.filename,
                delimiter = ',',
                skip_header = begin_line + 2,
                skip_footer = footer,
                )
        except(ValueError):
            # carrage returns require a doubling of skiplines
            data = npy.genfromtxt(
                self.filename,
                delimiter = ',',
                skip_header = (begin_line + 2)*2,
                skip_footer = footer,
                )

        # pna uses unicode coding for degree symbol, but we dont need that
        header = header.replace('\xb0','deg').rstrip('\n').rstrip('\r')

        return header, comments, data

    @property
    def frequency(self):
        '''
        Frequency object : :class:`~skrf.frequency.Frequency`
        '''
        header, comments, d = self.header, self.comments, self.data
        #try to pull out frequency unit
        cols = self.columns
        try:
            f_unit = cols[0].split('(')[1].split(')')[0]
        except:
            f_unit = 'hz'

        f = d[:,0]
        return Frequency.from_f(f, unit = f_unit)

    @property
    def n_traces(self):
        '''
        number of data traces : int
        '''
        return   self.data.shape[1] - 1

    @property
    def columns(self):
        '''
        List of column names : list of str

        This function is needed because Agilent allows the delimiter
        of a csv file (ie `'`) to be present in the header name. ridiculous.

        If splitting the header fails, then a suitable list is returned of
        the correct length, which looks like
         * ['Freq(?)','filename-0','filename-1',..]
        '''
        header,  d = self.header, self.data

        n_traces =  d.shape[1] - 1 # because theres is one frequency column

        if header.count(',') == n_traces:
            cols = header.split(',') # column names
        else:
            # the header contains too many delimiters. what loosers. maybe
            # we can split it on  `)'` instead
            if header.count('),') == n_traces:
                cols = header.split('),')
                # we need to add back the paranthesis we split on to all but
                # last columns
                cols =  [col + ')'  for col in cols[:-1]] + [cols[-1]]
            else:
                # I dont know how to seperate column names
                warn('Cant decipher header, so I\'m creating one. check output. ')
                cols = ['Freq(?),']+['%s-%i'%(util.basename_noext(filename),k) \
                    for k in range(n_traces)]
        return cols

    @property
    def scalar_networks(self):
        '''
        Returns list of Networks for each column : list

        the data is stored in the Network's `.s`  property, so its up
        to you to interpret results. if 'db' is in the column name then
        it is converted to linear before being store into `s`.


        Returns
        --------
        out : list of :class:`~skrf.network.Network` objects
            list of Networks representing the data contained in each column

        '''
        header, comments, d = self.header, self.comments, self.data
        n_traces =  d.shape[1] - 1 # because theres is one frequency column
        cols = self.columns
        freq = self.frequency

        # loop through columns and create a single network for each column
        ntwk_list = []
        for k in range(1,n_traces+1):
            s = d[:,k]
            if 'db' in cols[k].lower():
                s = mf.db_2_mag(s)

            ntwk_list.append(
                Network(
                    frequency = freq, s = s,comments = comments,
                    name = cols[k],*self.args, **self.kwargs)
                )

        return ntwk_list

    @property
    def networks(self):
        '''
        Reads a PNAX csv file, and returns a list of one-port Networks

        Note this only works if csv is save in Real/Imaginary format for now

        Parameters
        -----------
        filename : str
            filename

        Returns
        --------
        out : list of :class:`~skrf.network.Network` objects
            list of Networks representing the data contained in column pairs

        '''
        names = self.columns
        header, comments, d= self.header,self.comments, self.data

        ntwk_list = []
        if (self.n_traces)//2 == 0 : # / --> // for Python3 compatibility
            # this isnt complex data
            return self.scalar_networks
        else:
            for k in range((self.n_traces)//2):

                name = names[k*2+1]
                #print(names[k], names[k+1])
                if 'db' in names[k].lower() and 'deg' in names[k+1].lower():
                    s = mf.dbdeg_2_reim(d[:,k*2+1], d[:,k*2+2])
                elif 'real' in names[k].lower() and 'imag' in names[k+1].lower():
                    s = d[:,k*2+1]+1j*d[:,k*2+2]
                else:
                    warn('CSV format unrecognized in "%s" or "%s". It\'s up to you to intrepret the resultant network correctly.' % (names[k], names[k+1]))
                    s = d[:,k*2+1]+1j*d[:,k*2+2]

                ntwk_list.append(
                    Network(frequency = self.frequency, s=s, name=name,
                        comments=comments, *self.args, **self.kwargs)
                    )

        return ntwk_list

    @property
    def dict(self):
        '''
        '''
        return { self.columns[k]:self.data[:,k] \
            for k in range(self.n_traces+1)}
    @property
    def dataframe(self):
        '''
        Pandas DataFrame representation of csv file

        obviously this requires pandas
        '''
        from pandas import  Index, DataFrame

        index = Index(
            self.frequency.f_scaled,
            name = 'Frequency(%s)'%self.frequency.unit)

        return DataFrame(
                { self.columns[k]:self.data[:,k] \
                    for k in range(1,self.n_traces+1)},
                index=index,
                )

def pna_csv_header_split(filename):
    '''
    Split a Agilent csv file's header into a list

    This function is needed because Agilent allows the delimiter
    of a csv file (ie `'`) to be present in the header name. ridiculous.

    If splitting the header fails, then a suitable list is returned of
    the correct length, which looks like
     * ['Freq(?)','filename-0','filename-1',..]

    Parameters
    ------------
    filename : str
        csv filename

    Returns
    --------
    cols : list of str's
        list of column names
    '''
    warn("deprecated", DeprecationWarning)
    header, comments, d = read_pna_csv(filename)

    n_traces =  d.shape[1] - 1 # because theres is one frequency column

    if header.count(',') == n_traces:
        cols = header.split(',') # column names
    else:
        # the header contains too many delimiters. what loosers. maybe
        # we can split it on  `)'` instead
        if header.count('),') == n_traces:
            cols = header.split('),')
            # we need to add back the paranthesis we split on to all but
            # last columns
            cols =  [col + ')'  for col in cols[:-1]] + [cols[-1]]
        else:
            # i dont know how to seperate column names
            warn('Cant decipher header, so im creating one. check output. ')
            cols = ['Freq(?),']+['%s-%i'%(util.basename_noext(filename),k) \
                for k in range(n_traces)]
    return cols

def pna_csv_2_ntwks(filename):
    '''
    Reads a PNAX csv file, and returns a list of one-port Networks

    Note this only works if csv is save in Real/Imaginary format for now

    Parameters
    -----------
    filename : str
        filename

    Returns
    --------
    out : list of :class:`~skrf.network.Network` objects
        list of Networks representing the data contained in column pairs

    '''
    warn("deprecated", DeprecationWarning)
    #TODO: check the data's format (Real-imag or db/angle , ..)
    header, comments, d = read_pna_csv(filename)
    #import pdb;pdb.set_trace()

    names = pna_csv_header_split(filename)

    ntwk_list = []


    if (d.shape[1]-1)/2 == 0 :
        # this isnt complex data
        f = d[:,0]*1e-9
        if 'db' in header.lower():
            s = mf.db_2_mag(d[:,1])
        else:
            raise (NotImplementedError)
        name = os.path.splitext(os.path.basename(filename))[0]
        return Network(f=f, s=s, name=name, comments=comments)
    else:
        for k in range((d.shape[1]-1)/2):
            f = d[:,0]*1e-9
            name = names[k]
            print((names[k], names[k+1]))
            if 'db' in names[k].lower() and 'deg' in names[k+1].lower():
                s = mf.dbdeg_2_reim(d[:,k*2+1], d[:,k*2+2])
            elif 'real' in names[k].lower() and 'imag' in names[k+1].lower():
                s = d[:,k*2+1]+1j*d[:,k*2+2]
            else:
                print('WARNING: csv format unrecognized. ts up to you to  intrepret the resultant network correctly.')
                s = d[:,k*2+1]+1j*d[:,k*2+2]

            ntwk_list.append(
                Network(f=f, s=s, name=name, comments=comments)
                )

    return ntwk_list

def pna_csv_2_freq(filename):
    warn("deprecated", DeprecationWarning)
    header, comments, d = read_pna_csv(filename)
    #try to pull out frequency unit
    cols = pna_csv_header_split(filename)
    try:
        f_unit = cols[0].split('(')[1].split(')')[0]
    except:
        f_unit = 'hz'

    f = d[:,0]
    return Frequency.from_f(f, unit = f_unit)


def pna_csv_2_scalar_ntwks(filename, *args, **kwargs):
    '''
    Reads a PNAX csv file containing scalar traces, returning Networks



    Parameters
    -----------
    filename : str
        filename

    Returns
    --------
    out : list of :class:`~skrf.network.Network` objects
        list of Networks representing the data contained in column pairs

    '''
    warn("deprecated", DeprecationWarning)
    header, comments, d = read_pna_csv(filename)

    n_traces =  d.shape[1] - 1 # because theres is one frequency column

    cols = pna_csv_header_split(filename)


    #try to pull out frequency unit
    try:
        f_unit = cols[0].split('(')[1].split(')')[0]
    except:
        f_unit = 'hz'

    f = d[:,0]
    freq = Frequency.from_f(f, unit = f_unit)

    # loop through columns and create a single network for each column
    ntwk_list = []
    for k in range(1,n_traces+1):
        s = d[:,k]
        if 'db' in cols[k].lower():
            s = mf.db_2_mag(s)

        ntwk_list.append(
            Network(
                frequency = freq, s = s,comments = comments,
                name = cols[k],*args, **kwargs)
            )



    return ntwk_list




def read_zva_dat(filename, *args, **kwargs):
    '''
    Reads data from a dat file written by a R&S ZVA in dB/deg or re/im format

    This function returns a triplet containing header, comments and data.


    Parameters
    -------------
    filename : str
        the file
    \*args, \*\*kwargs :

    Returns
    ---------
    header : str
        The header string, which is the line following the 'BEGIN'
    data : :class:`numpy.ndarray`
        An array containing the data. The meaning of which depends on
        the header.

    '''
    #warn("deprecated", DeprecationWarning)
    with open(filename,'r') as fid:
        begin_line = -2
        comments = ''
        for k,line in enumerate(fid.readlines()):
            if line.startswith('%'):
                comments += line[1:]
                header = line
                begin_line = k+1

    data = npy.genfromtxt(
        filename,
        delimiter = ',',
        skip_header = begin_line,
        *args, **kwargs
        )

    return header, comments, data

def zva_dat_2_ntwks(filename):
    '''
    Read a dat file exported from a R&S ZVA in dB/deg or re/im format

    Parameters
    --------------
    filename : str
        full path or filename

    Returns
    ---------
    out : n
        2-Port Network

    Examples
    ----------

    '''
    header, comments, d = read_zva_dat(filename)
    col_headers = header.split(',')

    # set impedance to 50 Ohm (doesn't matter for now)
    z0 = npy.ones((npy.shape(d)[0]))*50
    # read f values, convert to GHz
    f = d[:,0]/1e9

    name = os.path.splitext(os.path.basename(filename))[0]

    if 're' in header.lower() and 'im' in header.lower():
        # this is a cvs in re/im format
        # -> no conversion required
        s = npy.zeros((len(f),2,2), dtype=complex)

        for k, h in enumerate(col_headers):
            if 's11' in h.lower() and 're' in h.lower():
                s[:,0,0] = d[:,k] + 1j*d[:,k+1]
            elif 's21' in h.lower() and 're' in h.lower():
                s[:,1,0] = d[:,k] + 1j*d[:,k+1]
            elif 's12' in h.lower() and 're' in h.lower():
                s[:,0,1] = d[:,k+1] #+ 1j*d[:,k+2]
            elif 's22' in h.lower() and 're' in h.lower():
                s[:,1,1] = d[:,k+1] #+ 1j*d[:,k+2]

    elif 'db' in header.lower() and not 'deg' in header.lower():
        # this is a cvs in db format (no deg values)
        # -> conversion required
        s = npy.zeros((len(f),2,2), dtype=complex)

        for k, h in enumerate(col_headers):
            # this doesn't always work! (depends on no. of channels, sequence of adding traces etc.
            # -> Needs changing!
            if 's11' in h.lower() and 'db' in h.lower():
                s[:,0,0] = mf.dbdeg_2_reim(d[:,k], d[:,k+2])
            elif 's21' in h.lower() and 'db' in h.lower():
                s[:,1,0] = mf.dbdeg_2_reim(d[:,k], d[:,k+2])

        n = Network(f=f,s=s,z0=z0, name = name)
        return n

    else:
        warn("File does not seem to be formatted properly (dB/deg or re/im)")

def read_all_zva_dat(dir='.', contains = None):
    '''
    Read all DAT files in a directory (from R&S ZVA)

    Parameters
    --------------
    dir : str, optional
        the directory to load from, default  \'.\'
    contains : str, optional
        if not None, only files containing this substring will be loaded

    Returns
    ---------
    out : dictionary
        dictionary containing all loaded DAT objects. keys are the
        filenames without extensions, and the values are the objects


    Examples
    ----------


    See Also
    ----------

    '''

    out={}
    for filename in os.listdir(dir):
        if contains is not None and contains not in filename:
            continue
        fullname = os.path.join(dir,filename)
        keyname = os.path.splitext(filename)[0]
        try:
            out[keyname] = zva_dat_2_ntwks(fullname)
            continue
        except:
            pass

        try:
            out[keyname] = Network(fullname)
            continue
        except:
            pass

    return out



def read_vectorstar_csv(filename, *args, **kwargs):
    '''
    Reads data from a csv file written by an Anritsu VectorStar

    Parameters
    -------------
    filename : str
        the file
    \*args, \*\*kwargs :

    Returns
    ---------
    header : str
        The header string, which is the line just before the data
    comments : str
        All lines that begin with a '!'
    data : :class:`numpy.ndarray`
        An array containing the data. The meaning of which depends on
        the header.


    '''
    with open(filename,'r') as fid:
        comments = ''.join([line for line in fid if line.startswith('!')])
        fid.seek(0)
        header = [line for line in fid if line.startswith('PNT')]
        fid.close()
        data = npy.genfromtxt(
            filename,
            comments='!',
            delimiter =',',
            skip_header = 1)[1:]
        comments = comments.replace('\r','')
        comments = comments.replace('!','')

    return header, comments, data

def vectorstar_csv_2_ntwks(filename):
    '''
    Reads a vectorstar csv file, and returns a list of one-port Networks

    Note this only works if csv is save in Real/Imaginary format for now

    Parameters
    -----------
    filename : str
        filename

    Returns
    --------
    out : list of :class:`~skrf.network.Network` objects
        list of Networks representing the data contained in column pairs

    '''
    #TODO: check the data's format (Real-imag or db/angle , ..)
    header, comments, d = read_vectorstar_csv(filename)
    names = [line for line in comments.split('\n') \
        if line.startswith('PARAMETER')][0].split(',')[1:]


    return [Network(
        f = d[:,k*3+1],
        s = d[:,k*3+2] + 1j*d[:,k*3+3],
        z0 = 50,
        name = names[k].rstrip(),
        comments = comments,
        ) for k in range(d.shape[1]/3)]


