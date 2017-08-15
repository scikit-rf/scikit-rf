
'''
.. module:: skrf.io.general
========================================
general (:mod:`skrf.io.general`)
========================================

General io functions for reading and writing skrf objects

.. autosummary::
    :toctree: generated/

    read
    read_all
    read_all_networks
    write
    write_all
    save_sesh


Writing output to spreadsheet

.. autosummary::
    :toctree: generated/

    network_2_spreadsheet
    networkset_2_spreadsheet


'''
import sys

import six.moves.cPickle as pickle
from six.moves.cPickle import UnpicklingError


import inspect
import os
import zipfile
import warnings
import sys

from ..util import get_extn, get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import  Media
from ..networkSet import NetworkSet
from ..calibration.calibration import Calibration

from copy import copy
dir_ = copy(dir)

# delayed import: from pandas import DataFrame, Series for ntwk_2_spreadsheet

# file extension conventions for skrf objects.
global OBJ_EXTN
OBJ_EXTN = [
    [Frequency, 'freq'],
    [Network, 'ntwk'],
    [NetworkSet, 'ns'],
    [Calibration, 'cal'],
    [Media, 'med'],
    [object, 'p'],
    ]


def read(file, *args, **kwargs):
    '''
    Read  skrf object[s] from a pickle file

    Reads a skrf object that is written with :func:`write`, which uses
    the :mod:`pickle` module.

    Parameters
    ------------
    file : str or file-object
        name of file, or  a file-object
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load

    Examples
    -------------
    >>> n = rf.Network(f=[1,2,3],s=[1,1,1],z0=50)
    >>> n.write('my_ntwk.ntwk')
    >>> n_2 = rf.read('my_ntwk.ntwk')

    See Also
    ----------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory

    Notes
    -------
    if `file` is a file-object it is left open, if it is a filename then
    a file-object is opened and closed. If file is a file-object
    and reading fails, then the position is reset back to 0 using seek
    if possible.
    '''
    fid = get_fid(file, mode='rb')
    try:
        obj = pickle.load(fid, *args, **kwargs)
    except (UnpicklingError, UnicodeDecodeError) as e:
        # if fid is seekable then reset to beginning of file
        fid.seek(0)

        if isinstance(file, str):
            # we created the fid so close it
            fid.close()
        raise

    if isinstance(file, str):
        # we created the fid so close it
        fid.close()

    return obj

def write(file, obj, overwrite = True):
    '''
    Write skrf object[s] to a file

    This uses the :mod:`pickle` module to write skrf objects to a file.
    Note that you can write any pickl-able python object. For example,
    you can write a list or dictionary of :class:`~skrf.network.Network`
    objects
    or :class:`~skrf.calibration.calibration.Calibration` objects. This
    will write out a single file. If you would like to write out a
    seperate file for each object, use :func:`write_all`.

    Parameters
    ------------
    file : file or string
        File or filename to which the data is saved.  If file is a
        file-object, then the filename is unchanged.  If file is a
        string, an appropriate extension will be appended to the file
        name if it does not already have an extension.

    obj : an object, or list/dict of objects
        object or list/dict of objects to write to disk

    overwrite : Boolean
        if file exists, should it be overwritten?

    Notes
    -------

    If `file` is a str, but doesnt contain a suffix, one is chosen
    automatically. Here are the extensions


    ====================================================  ===============
    skrf object                                           extension
    ====================================================  ===============
    :class:`~skrf.frequency.Frequency`                    '.freq'
    :class:`~skrf.network.Network`                        '.ntwk'
    :class:`~skrf.networkSet.NetworkSet`                  '.ns'
    :class:`~skrf.calibration.calibration.Calibration`    '.cal'
    :class:`~skrf.media.media.Media`                      '.med'
    other                                                 '.p'
    ====================================================  ===============

    To make the file written by this method cross-platform, the pickling
    protocol 2 is used. See :mod:`pickle` for more info.

    Examples
    -------------
    Convert a touchstone file to a pickled Network,

    >>> n = rf.Network('my_ntwk.s2p')
    >>> rf.write('my_ntwk',n)
    >>> n_red = rf.read('my_ntwk.ntwk')

    Writing a list of different objects

    >>> n = rf.Network('my_ntwk.s2p')
    >>> ns = rf.NetworkSet([n,n,n])
    >>> rf.write('out',[n,ns])
    >>> n_red = rf.read('out.p')

    See Also
    ------------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory
    skrf.network.Network.write : write method of Network
    skrf.calibration.calibration.Calibration.write : write method of Calibration


    '''
    if isinstance(file, str):
        extn = get_extn(file)
        if extn is None:
            # if there is not extension add one
            for obj_extn in OBJ_EXTN:
                if isinstance(obj, obj_extn[0]):
                    extn = obj_extn[1]
                    break
            file = file + '.' + extn

        if os.path.exists(file):
            if not overwrite:
                warnings.warn('file exists, and overwrite option is False. Not writing.')
                return

        with open(file, 'wb') as fid:
            pickle.dump(obj, fid, protocol=2)

    else:
        fid = file
        pickle.dump(obj, fid, protocol=2)
        fid.close()

def read_all(dir='.', contains = None, f_unit = None, obj_type=None):
    '''
    Read all skrf objects in a directory


    Attempts to load all files in `dir`, using :func:`read`. Any file
    that is not readable by skrf is skipped. Optionally, simple filtering
    can be achieved through the use of `contains` argument.

    Parameters
    --------------
    dir : str, optional
        the directory to load from, default  \'.\'
    contains : str, optional
        if not None, only files containing this substring will be loaded
    f_unit : ['hz','khz','mhz','ghz','thz']
        for all :class:`~skrf.network.Network` objects, set their
        frequencies's :attr:`~skrf.frequency.Frequency.f_unit`
    obj_type : str
        Name of skrf object types to read (ie 'Network')

    Returns
    ---------
    out : dictionary
        dictionary containing all loaded skrf objects. keys are the
        filenames without extensions, and the values are the objects


    Examples
    ----------
    >>> rf.read_all('skrf/data/')
    {'delay_short': 1-Port Network: 'delay_short',  75-110 GHz, 201 pts, z0=[ 50.+0.j],
    'line': 2-Port Network: 'line',  75-110 GHz, 201 pts, z0=[ 50.+0.j  50.+0.j],
    'ntwk1': 2-Port Network: 'ntwk1',  1-10 GHz, 91 pts, z0=[ 50.+0.j  50.+0.j],
    'one_port': one port Calibration: 'one_port', 500-750 GHz, 201 pts, 4-ideals/4-measured,
    ...

    >>> rf.read_all('skrf/data/', obj_type = 'Network')
    {'delay_short': 1-Port Network: 'delay_short',  75-110 GHz, 201 pts, z0=[ 50.+0.j],
    'line': 2-Port Network: 'line',  75-110 GHz, 201 pts, z0=[ 50.+0.j  50.+0.j],
    'ntwk1': 2-Port Network: 'ntwk1',  1-10 GHz, 91 pts, z0=[ 50.+0.j  50.+0.j],
    ...

    See Also
    ----------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory
    '''

    out={}
    for filename in os.listdir(dir):
        if contains is not None and contains not in filename:
            continue
        fullname = os.path.join(dir,filename)
        keyname = os.path.splitext(filename)[0]
        try:
            out[keyname] = read(fullname)
            continue
        except:
            pass

        try:
            out[keyname] = Network(fullname)
            continue
        except:
            pass

    if f_unit is not None:
        for keyname in out:
            try:
                out[keyname].frequency.unit = f_unit
            except:
                pass

    if obj_type is not None:
        out = dict([(k, out[k]) for k in out if
            isinstance(out[k],sys.modules[__name__].__dict__[obj_type])])

    return out


def read_all_networks(*args, **kwargs):
    '''
    Read all networks in a directory.

    This is a convenience function. It just calls::

        read_all(*args,obj_type='Network', **kwargs)

    See Also
    ----------
    read_all
    '''
    if 'f_unit' not in kwargs:
        kwargs.update({'f_unit':'ghz'})
    return read_all(*args,obj_type='Network', **kwargs)

ran = read_all_networks

def write_all(dict_objs, dir='.', *args, **kwargs):
    '''
    Write a dictionary of skrf objects individual files in `dir`.

    Each object is written to its own file. The filename used for each
    object is taken from its key in the dictionary. If no extension
    exists in the key, then one is added. See :func:`write` for a list
    of extensions. If you would like to write the dictionary to a single
    output file use :func:`write`.

    Notes
    -------
    Any object in  dict_objs that is pickl-able will be written.


    Parameters
    ------------
    dict_objs : dict
        dictionary of skrf objects
    dir : str
        directory to save skrf objects into
    \*args, \*\*kwargs :
            passed through to :func:`~skrf.io.general.write`. `overwrite`
            option may be of use.

    See Also
    -----------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory

    Examples
    ----------
    Writing a diction of different skrf objects

    >>> from skrf.data import line, short
    >>> d = {'ring_slot':ring_slot, 'one_port_cal':one_port_cal}
    >>> rf.write_all(d)

    '''
    if not os.path.exists('.'):
        raise OSError('No such directory: %s'%dir)


    for k in dict_objs:
        filename = k
        obj = dict_objs[k]

        extn = get_extn(filename)
        if extn is None:
            # if there is not extension add one
            for obj_extn in OBJ_EXTN:
                if isinstance(obj, obj_extn[0]):
                    extn = obj_extn[1]
                    break
            filename = filename + '.' + extn
        try:
            with open(os.path.join(dir+'/', filename), 'wb') as fid:
                write(fid, obj,*args, **kwargs)
        except Exception as inst:
            print(inst)
            warnings.warn('couldnt write %s: %s'%(k,str(inst)))

            pass

def save_sesh(dict_objs, file='skrfSesh.p', module='skrf', exclude_prefix='_'):
    '''
    Save all `skrf` objects in the local namespace.

    This is used to save current workspace in a hurry, by passing it the
    output of :func:`locals` (see Examples). Note this can be
    used for other modules as well by passing a different `module` name.

    Parameters
    ------------
    dict_objs : dict
        dictionary containing `skrf` objects. See the Example.
    file : str or file-object, optional
        the file to save all objects to
    module : str, optional
        the module name to grep for.
    exclude_prefix: str, optional
        dont save objects which have this as a prefix.

    See Also
    ----------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory


    Examples
    ---------
    Write out all skrf objects in current namespace.

    >>> rf.write_all(locals(), 'mysesh.p')


    '''
    objects = {}
    print('pickling: ')
    for k in dict_objs:
        try:
            if module  in inspect.getmodule(dict_objs[k]).__name__:
                try:
                    pickle.dumps(dict_objs[k])
                    if k[0] != '_':
                        objects[k] = dict_objs[k]
                        print(k+', ')
                finally:
                    pass

        except(AttributeError, TypeError):
            pass
    if len (objects ) == 0:
        print('nothing')

    write(file, objects)

def load_all_touchstones(dir = '.', contains=None, f_unit=None):
    '''
    Loads all touchtone files in a given dir into a dictionary.

    Notes
    -------
    Alternatively you can use the :func:`read_all` function.

    Parameters
    -----------
    dir :   string
            the path
    contains :      string
            a string the filenames must contain to be loaded.
    f_unit  : ['hz','mhz','ghz']
            the frequency unit to assign all loaded networks. see
            :attr:`frequency.Frequency.unit`.

    Returns
    ---------
    ntwkDict : a dictonary with keys equal to the file name (without
            a suffix), and values equal to the corresponding ntwk types

    Examples
    ----------
    >>> ntwk_dict = rf.load_all_touchstones('.', contains ='20v')

    See Also
    -----------
    read_all
    '''
    ntwkDict = {}

    for f in os.listdir (dir):
        if contains is not None and contains not in f:
            continue
        fullname = os.path.join(dir,f)
        keyname,extn = os.path.splitext(f)
        extn = extn.lower()
        try:
            if extn[1]== 's' and extn[-1]=='p':
                ntwkDict[keyname]=(Network(dir +'/'+f))
                if f_unit is not None: ntwkDict[keyname].frequency.unit=f_unit
        except:
            pass
    return ntwkDict

def write_dict_of_networks(ntwkDict, dir='.'):
    '''
    Saves a dictionary of networks touchstone files in a given directory

    The filenames assigned to the touchstone files are taken from
    the keys of the dictionary.

    Parameters
    -----------
    ntwkDict : dictionary
            dictionary of :class:`Network` objects
    dir : string
            directory to write touchstone file to


    '''
    warnings.warn('Deprecated. use write_all.', DeprecationWarning)
    for ntwkKey in ntwkDict:
        ntwkDict[ntwkKey].write_touchstone(filename = dir+'/'+ntwkKey)

def read_csv(filename):
    '''
    Read a 2-port s-parameter data from a csv file.

    Specifically, this reads a two-port csv file saved from a Rohde Shcwarz
    ZVA-40, and possibly other network analyzers. It returns into a
    :class:`Network` object.

    Parameters
    ------------
    filename : str
            name of file

    Returns
    --------
    ntwk : :class:`Network` object
            the network representing data in the csv file
    '''

    ntwk = Network(name=filename[:-4])
    try:
        data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(9))
        s11 = data[:,1] +1j*data[:,2]
        s21 = data[:,3] +1j*data[:,4]
        s12 = data[:,5] +1j*data[:,6]
        s22 = data[:,7] +1j*data[:,8]
        ntwk.s = npy.array([[s11, s21],[s12,s22]]).transpose().reshape(-1,2,2)
    except(IndexError):
        data = npy.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(3))
        ntwk.s = data[:,1] +1j*data[:,2]

    ntwk.frequency.f = data[:,0]
    ntwk.frequency.unit='ghz'

    return ntwk


## file conversion
def statistical_2_touchstone(file_name, new_file_name=None,\
        header_string='# GHz S RI R 50.0'):
    '''
    Converts Statistical file to a touchstone file.

    Converts the file format used by Statistical and other Dylan Williams
    software to standard touchstone format.

    Parameters
    ------------
    file_name : string
            name of file to convert
    new_file_name : string
            name of new file to write out (including extension)
    header_string : string
            touchstone header written to first beginning of file

    '''
    if new_file_name is None:
        new_file_name = 'tmp-'+file_name
        remove_tmp_file = True

    # This breaks compatibility with python 2.6 and older
    with file(file_name, 'r') as old_file, open(new_file_name, 'w') as new_file: 
        new_file.write('%s\n'%header_string)
        for line in old_file:
            new_file.write(line)

    if remove_tmp_file is True:
        os.rename(new_file_name,file_name)

def network_2_spreadsheet(ntwk, file_name =None, file_type= 'excel', form='db',
    *args, **kwargs):
    '''
    Write a Network object to a spreadsheet, for your boss

    Write the s-parameters  of a network to a spreadsheet, in a variety
    of forms. This functions makes use of the pandas module, which in
    turn makes use of the xlrd module. These are imported during this
    function call. For more details about the file-writing functions
    see the pandas.DataFrom.to_?? functions.

    Notes
    ------
    The frequency unit used in the spreadsheet is take from
    `ntwk.frequency.unit`

    Parameters
    -----------
    ntwk :  :class:`~skrf.network.Network` object
        the network to write
    file_name : str, None
        the file_name to write. if None,  ntwk.name is used.
    file_type : ['csv','excel','html']
        the type of file to write. See pandas.DataFrame.to_??? functions.
    form : 'db','ma','ri'
        format to write data,
        * db = db, deg
        * ma = mag, deg
        * ri = real, imag
    \*args, \*\*kwargs :
        passed to pandas.DataFrame.to_??? functions.


    See Also
    ---------
    networkset_2_spreadsheet : writes a spreadsheet for many networks
    '''
    from pandas import DataFrame, Series # delayed because its not a requirement
    file_extns = {'csv':'csv','excel':'xls','html':'html'}

    form = form.lower()
    if form not in ['db','ri','ma']:
        raise ValueError('`form` must be either `db`,`ma`,`ri`')


    file_type = file_type.lower()
    if file_type not in file_extns.keys():
        raise ValueError('file_type must be `csv`,`html`,`excel` ')
    if ntwk.name is None and file_name is None:
        raise ValueError('Either ntwk must have name or give a file_name')


    if file_name is None and 'excel_writer' not in kwargs.keys():
        file_name = ntwk.name + '.'+file_extns[file_type]

    d = {}
    index =ntwk.frequency.f_scaled

    if form =='db':
        for m,n in ntwk.port_tuples:
            d['S%i%i Log Mag(dB)'%(m+1,n+1)] = \
                Series(ntwk.s_db[:,m,n], index = index)
            d[u'S%i%i Phase(deg)'%(m+1,n+1)] = \
                Series(ntwk.s_deg[:,m,n], index = index)
    elif form =='ma':
        for m,n in ntwk.port_tuples:
            d['S%i%i Mag(lin)'%(m+1,n+1)] = \
                Series(ntwk.s_mag[:,m,n], index = index)
            d[u'S%i%i Phase(deg)'%(m+1,n+1)] = \
                Series(ntwk.s_deg[:,m,n], index = index)
    elif form =='ri':
        for m,n in ntwk.port_tuples:
            d['S%i%i Real'%(m+1,n+1)] = \
                Series(ntwk.s_re[:,m,n], index = index)
            d[u'S%i%i Imag'%(m+1,n+1)] = \
                Series(ntwk.s_im[:,m,n], index = index)

    df = DataFrame(d)
    df.__getattribute__('to_%s'%file_type)(file_name,
        index_label='Freq(%s)'%ntwk.frequency.unit, *args, **kwargs)

def network_2_dataframe(ntwk, attrs=['s_db'], ports = None):
    '''
    Convert one or more attributes of a network to a pandas DataFrame

    Parameters
    --------------
    ntwk :  :class:`~skrf.network.Network` object
        the network to write
    attrs : list Network attributes
        like ['s_db','s_deg']
    ports : list of tuples
        list of port pairs to write. defaults to ntwk.port_tuples
        (like [[0,0]])

    Returns
    ----------
    df : pandas DataFrame Object
    '''
    from pandas import DataFrame, Series # delayed because its not a requirement
    d = {}
    index =ntwk.frequency.f_scaled

    if ports is None:
        ports = ntwk.port_tuples

    for attr in attrs:
        for m,n in ports:
            d['%s %i%i'%(attr, m+1,n+1)] = \
                Series(ntwk.__getattribute__(attr)[:,m,n], index = index)

    return DataFrame(d)

def networkset_2_spreadsheet(ntwkset, file_name=None, file_type= 'excel',
    *args, **kwargs):
    '''
    Write a NetworkSet object to a spreadsheet, for your boss

    Write  the s-parameters  of a each network in the networkset to a
    spreadsheet. If the `excel` file_type is used, then each network,
    is written to its own sheet, with the sheetname taken from the
    network `name` attribute.
    This functions makes use of the pandas module, which in turn makes
    use of the xlrd module. These are imported during this function

    Notes
    ------
    The frequency unit used in the spreadsheet is take from
    `ntwk.frequency.unit`

    Parameters
    -----------
    ntwkset :  :class:`~skrf.networkSet.NetworkSet` object
        the network to write
    file_name : str, None
        the file_name to write. if None,  ntwk.name is used.
    file_type : ['csv','excel','html']
        the type of file to write. See pandas.DataFrame.to_??? functions.
    form : 'db','ma','ri'
        format to write data,
        * db = db, deg
        * ma = mag, deg
        * ri = real, imag
    \*args, \*\*kwargs :
        passed to pandas.DataFrame.to_??? functions.


    See Also
    ---------
    networkset_2_spreadsheet : writes a spreadsheet for many networks
    '''
    from pandas import DataFrame, Series, ExcelWriter # delayed because its not a requirement
    if ntwkset.name is None and file_name is None:
        raise(ValueError('Either ntwkset must have name or give a file_name'))

    if file_type == 'excel':
        writer = ExcelWriter(file_name)
        [network_2_spreadsheet(k, writer, sheet_name =k.name, *args, **kwargs) for k in ntwkset]
        writer.save()
    else:
        [network_2_spreadsheet(k,*args, **kwargs) for k in ntwkset]


# Provide a StringBuffer that let's me work with Python2 strings and Python3 unicode strings without thinking
if sys.version_info < (3, 0):
    import StringIO

    class StringBuffer(StringIO.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
else:
    import io
    StringBuffer = io.StringIO
