
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
    fid = open(filename,'r')
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
    
    fid.close()
    
    try:
        data = npy.genfromtxt(
            filename, 
            delimiter = ',',
            skip_header = begin_line + 2,
            skip_footer = footer,
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
        fid = open(self.filename, 'r')
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
        
        fid.close()
        
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
        of a csv file (ie `'`) to be present in the header name. rediculous.
        
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
                # i dont know how to seperate column names
                warn('Cant decipher header, so im creating one. check output. ')
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
        if (self.n_traces)/2 == 0 :
            # this isnt complex data
            return self.scalar_networks
        else:
            for k in range((self.n_traces)/2):
                
                name = names[k*2+1]
                #print(names[k], names[k+1])
                if 'db' in names[k].lower() and 'deg' in names[k+1].lower():
                    s = mf.dbdeg_2_reim(d[:,k*2+1], d[:,k*2+2])
                elif 'real' in names[k].lower() and 'imag' in names[k+1].lower():
                    s = d[:,k*2+1]+1j*d[:,k*2+2]
                else:
                    warn('CSV format unrecognized. its up to you to  intrepret the resultant network correctly.')
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
    of a csv file (ie `'`) to be present in the header name. rediculous.
    
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
            print(names[k], names[k+1])
            if 'db' in names[k].lower() and 'deg' in names[k+1].lower():
                s = mf.dbdeg_2_reim(d[:,k*2+1], d[:,k*2+2])
            elif 'real' in names[k].lower() and 'imag' in names[k+1].lower():
                s = d[:,k*2+1]+1j*d[:,k*2+2]
            else:
                print ('WARNING: csv format unrecognized. ts up to you to  intrepret the resultant network correctly.')
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
    fid = open(filename,'r')
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
        

