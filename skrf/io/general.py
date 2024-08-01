"""
.. module:: skrf.io.general

========================================
general (:mod:`skrf.io.general`)
========================================

General input/output functions for reading and writing skrf objects


Pickle functions
------------------

The read/write methods use the pickle module. These should only be used
for temporary storage.

.. autosummary::
   :toctree: generated/

   read
   read_all
   read_all_networks
   write
   write_all
   save_sesh


Spreadsheets
-----------------------------

.. autosummary::
   :toctree: generated/

   network_2_spreadsheet
   networkset_2_spreadsheet

Pandas dataframe
----------------------------------

.. autosummary::
   :toctree: generated/

   network_2_dataframe

Statistics
----------

.. autosummary::
   :toctree: generated/

    statistical_2_touchstone

JSON
-------

.. autosummary::
   :toctree: generated/

   TouchstoneEncoder
   to_json_string
   from_json_string


"""
from __future__ import annotations

import glob
import inspect
import json
import os
import pickle
import sys
import warnings
from io import StringIO
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import numpy as np
from pandas import DataFrame, ExcelWriter, Series

from ..frequency import Frequency
from ..network import Network
from ..networkSet import NetworkSet
from ..util import get_extn, get_fid


def _get_extension(inst: Any) -> str:
    """File extension conventions for skrf objects.
    """
    from ..calibration.calibration import Calibration
    from ..media import Media

    extensions = [
        (Frequency, "freq"),
        (Network, "ntwk"),
        (NetworkSet, "ns"),
        (Calibration, "cal"),
        (Media, "med"),
    ]

    for cls, ext in extensions:
        print(cls, ext)
        if isinstance(inst, cls):
            return ext
    return "p"

def read(file, *args, **kwargs):
    r"""
    Read  skrf object[s] from a pickle file.

    Reads a skrf object that is written with :func:`write`, which uses
    the :mod:`pickle` module.

    Parameters
    ----------
    file : str, Path, or file-object
        name of file, or  a file-object
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load


    .. note::
        If `file` is a:

        * a file-object, it is left open

        * a filename, then a file-object is opened and closed.

        * a file-object and reading fails, then the position is reset back to 0 using seek if possible.


    Examples
    --------
    >>> n = rf.Network(f=[1,2,3],s=[1,1,1],z0=50)
    >>> n.write('my_ntwk.ntwk')
    >>> n_2 = rf.read('my_ntwk.ntwk')

    See Also
    --------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory
    """
    fid = get_fid(file, mode='rb')
    try:
        obj = pickle.load(fid, *args, **kwargs)
    except (UnpicklingError, UnicodeDecodeError):
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
    """
    Write skrf object[s] to a file.

    This uses the :mod:`pickle` module to write skrf objects to a file.
    Note that you can write any pickl-able python object. For example,
    you can write a list or dictionary of :class:`~skrf.network.Network`
    objects
    or :class:`~skrf.calibration.calibration.Calibration` objects. This
    will write out a single file. If you would like to write out a
    separate file for each object, use :func:`write_all`.

    Parameters
    ----------
    file : file, Path, or string
        File or filename to which the data is saved.  If file is a
        file-object, then the filename is unchanged.  If file is a
        string, an appropriate extension will be appended to the file
        name if it does not already have an extension.

    obj : an object, or list/dict of objects
        object or list/dict of objects to write to disk

    overwrite : Boolean
        if file exists, should it be overwritten?


    .. note::
        If `file` is a string, but doesnt contain a suffix, one is chosen
        automatically. Here are the extensions:


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

    .. note::
        To make the file written by this method cross-platform, the pickling
        protocol 2 is used. See :mod:`pickle` for more info.

    Examples
    --------
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
    --------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory
    skrf.network.Network.write : write method of Network
    skrf.calibration.calibration.Calibration.write : write method of Calibration


    """
    if isinstance(file, str):
        extn = get_extn(file)
        if extn is None:
            # if there is not extension add one
            file += f".{_get_extension(obj)}"

        if os.path.exists(file):
            if not overwrite:
                warnings.warn('file exists, and overwrite option is False. Not writing.', stacklevel=2)
                return

        with open(file, 'wb') as fid:
            pickle.dump(obj, fid, protocol=2)

    else:
        fid = file
        pickle.dump(obj, fid, protocol=2)
        fid.close()

def read_all(dir: str | Path = '.', sort = True, contains = None, f_unit = None,
        obj_type=None, files: list=None, recursive=False) -> dict:
    """
    Read all skrf objects in a directory.

    Attempts to load all files in `dir`, using :func:`read`. Any file
    that is not readable by skrf is skipped. Optionally, simple filtering
    can be achieved through the use of `contains` argument.

    Parameters
    ----------
    dir : str or Path, optional
        the directory to load from, default  \'.\'
    sort: boolean, default is True
        filenames sorted by https://docs.python.org/3/library/stdtypes.html#list.sort without arguements
    contains : str, optional
        if not None, only files containing this substring will be loaded
    f_unit : ['hz','khz','mhz','ghz','thz']
        for all :class:`~skrf.network.Network` objects, set their
        frequencies's :attr:`~skrf.frequency.Frequency.f_unit`
    obj_type : str
        Name of skrf object types to read (ie 'Network')
    files : list, optional
        list of files to load, bypasses dir parameter.
    recursive : bool, optional
        If True, search in the specified directory and all other nested directories

    Returns
    -------
    out : dictionary
        dictionary containing all loaded skrf objects. keys are the
        filenames without extensions, and the values are the objects


    Examples
    --------
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

    >>> rf.read_all(files = ['skrf/data/delay_short.s1p', 'skrf/data/line.s2p'], obj_type = 'Network')
    {'delay_short': 1-Port Network: 'delay_short',  75-110 GHz, 201 pts, z0=[ 50.+0.j],
    'line': 2-Port Network: 'line',  75-110 GHz, 201 pts, z0=[ 50.+0.j  50.+0.j]}

    See Also
    ----------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory
    """

    # Convert a Path object to a string
    dir = str(dir.resolve()) if isinstance(dir, Path) else dir

    out={}

    filelist = []
    if files is None:
        if recursive:
            if not dir.endswith(os.path.sep):
                dir += os.path.sep
            dir += '**'
        for filename in glob.iglob(os.path.join(dir, '*.s*p'), recursive=recursive):
            filelist.append(filename)
    else:
        filelist.extend(files)

    if sort is True:
        filelist.sort()

    for filename in filelist:
        if contains is not None and contains not in filename:
            continue
        fullname = filename
        keyname = os.path.splitext(filename.split(os.path.sep)[-1])[0]
        try:
            out[keyname] = read(fullname)
            continue
        except Exception:
            pass

        try:
            out[keyname] = Network(fullname)
            continue
        except Exception:
            pass

    if f_unit is not None:
        for keyname in out:
            try:
                out[keyname].frequency.unit = f_unit
            except Exception:
                pass

    if obj_type is not None:
        out = {k: out[k] for k in out if
            isinstance(out[k],sys.modules[__name__].__dict__[obj_type])}

    return out


def read_all_networks(*args, **kwargs):
    """
    Read all networks in a directory.

    This is a convenience function. It just calls::

        read_all(*args,obj_type='Network', **kwargs)

    See Also
    --------
    read_all
    """
    return read_all(*args,obj_type='Network', **kwargs)

ran = read_all_networks

def write_all(dict_objs, dir='.', *args, **kwargs):
    r"""
    Write a dictionary of skrf objects individual files in `dir`.

    Each object is written to its own file. The filename used for each
    object is taken from its key in the dictionary. If no extension
    exists in the key, then one is added. See :func:`write` for a list
    of extensions. If you would like to write the dictionary to a single
    output file use :func:`write`.


    .. note::
        Any object in  dict_objs that is pickl-able will be written.


    Parameters
    ----------
    dict_objs : dict
        dictionary of skrf objects
    dir : str
        directory to save skrf objects into
    \*args, \*\*kwargs :
            passed through to :func:`~skrf.io.general.write`. `overwrite`
            option may be of use.

    See Also
    --------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory

    Examples
    --------
    Writing a diction of different skrf objects

    >>> from skrf.data import line, short
    >>> d = {'ring_slot':ring_slot, 'one_port_cal':one_port_cal}
    >>> rf.write_all(d)

    """
    if not os.path.exists('.'):
        raise OSError(f'No such directory: {dir}')



    for k in dict_objs:
        filename = k
        obj = dict_objs[k]

        extn = get_extn(filename)
        if extn is None:
            # if there is not extension add one
            filename += f".{_get_extension(obj)}"
        try:
            with open(os.path.join(dir+'/', filename), 'wb') as fid:
                write(fid, obj,*args, **kwargs)
        except Exception as inst:
            print(inst)
            warnings.warn(f'couldnt write {k}: {inst}', stacklevel=2)

            pass

def save_sesh(dict_objs, file='skrfSesh.p', module='skrf', exclude_prefix='_'):
    """
    Save all `skrf` objects in the local namespace.

    This is used to save current workspace in a hurry, by passing it the
    output of :func:`locals` (see Examples). Note this can be
    used for other modules as well by passing a different `module` name.

    Parameters
    ----------
    dict_objs : dict
        dictionary containing `skrf` objects. See the Example.
    file : str or file-object, optional
        the file to save all objects to
    module : str, optional
        the module name to grep for.
    exclude_prefix: str, optional
        dont save objects which have this as a prefix.

    See Also
    --------
    read : read a skrf object
    write : write skrf object[s]
    read_all : read all skrf objects in a directory
    write_all : write dictionary of skrf objects to a directory


    Examples
    --------
    Write out all skrf objects in current namespace.

    >>> rf.write_all(locals(), 'mysesh.p')


    """
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
    """
    Loads all touchtone files in a given dir into a dictionary.

    Notes
    -------


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
    ntwkDict : a dictionary with keys equal to the file name (without
            a suffix), and values equal to the corresponding ntwk types

    Examples
    ----------
    >>> ntwk_dict = rf.load_all_touchstones('.', contains ='20v')

    See Also
    -----------
    read_all
    """
    ntwkDict = {}

    for f in os.listdir (dir):
        if contains is not None and contains not in f:
            continue
        keyname,extn = os.path.splitext(f)
        extn = extn.lower()
        try:
            if extn[1]== 's' and extn[-1]=='p':
                ntwkDict[keyname]=(Network(dir +'/'+f))
                if f_unit is not None:
                    ntwkDict[keyname].frequency.unit=f_unit
        except Exception:
            pass
    return ntwkDict

def write_dict_of_networks(ntwkDict, dir='.'):
    """
    Saves a dictionary of networks touchstone files in a given directory

    The filenames assigned to the touchstone files are taken from
    the keys of the dictionary.

    Parameters
    -----------
    ntwkDict : dictionary
            dictionary of :class:`Network` objects
    dir : string
            directory to write touchstone file to


    """
    warnings.warn('Deprecated. use write_all.', DeprecationWarning, stacklevel=2)
    for ntwkKey in ntwkDict:
        ntwkDict[ntwkKey].write_touchstone(filename = dir+'/'+ntwkKey)

def read_csv(filename):
    """
    Read a 2-port s-parameter data from a csv file.

    Specifically, this reads a two-port csv file saved from a Rohde Schwarz
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
    """

    ntwk = Network(name=filename[:-4])
    try:
        data = np.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(9))
        s11 = data[:,1] +1j*data[:,2]
        s21 = data[:,3] +1j*data[:,4]
        s12 = data[:,5] +1j*data[:,6]
        s22 = data[:,7] +1j*data[:,8]
        ntwk.s = np.array([[s11, s21],[s12,s22]]).transpose().reshape(-1,2,2)
    except(IndexError):
        data = np.loadtxt(filename, skiprows=3,delimiter=',',\
                usecols=range(3))
        ntwk.s = data[:,1] +1j*data[:,2]

    ntwk.frequency.f = data[:,0]

    return ntwk


## file conversion
def statistical_2_touchstone(file_name, new_file_name=None,\
        header_string='# GHz S RI R 50.0'):
    """
    Converts Statistical file to a touchstone file.

    Converts the file format used by Statistical and other Dylan Williams
    software to standard touchstone format.

    Parameters
    ----------
    file_name : string
            name of file to convert
    new_file_name : string
            name of new file to write out (including extension)
    header_string : string
            touchstone header written to first beginning of file

    """
    remove_tmp_file = new_file_name is None
    if remove_tmp_file:
        new_file_name = 'tmp-'+file_name

    # This breaks compatibility with python 2.6 and older
    with open(file_name) as old_file, open(new_file_name, 'w') as new_file:
        new_file.write(f'{header_string}\n')
        for line in old_file:
            new_file.write(line)

    if remove_tmp_file:
        os.rename(new_file_name,file_name)

def network_2_spreadsheet(ntwk: Network, file_name: str | Path = None,
        file_type: str = 'excel', form: str ='db', *args, **kwargs):
    r"""
    Write a Network object to a spreadsheet, for your boss.

    Write the s-parameters  of a network to a spreadsheet, in a variety
    of forms. This functions makes use of the pandas module, which in
    turn makes use of the xlrd module. These are imported during this
    function call. For more details about the file-writing functions
    see the `pandas.DataFrom.to_???` functions.


    .. note::
        The frequency unit used in the spreadsheet is take from
        `ntwk.frequency.unit`


    Parameters
    ----------
    ntwk :  :class:`~skrf.network.Network` object
        the network to write
    file_name : str, Path or None
        the file_name to write. if None,  ntwk.name is used.
    file_type : ['csv','excel','html']
        the type of file to write. See `pandas.DataFrame.to_???` functions.
    form : 'db','ma','ri'
        format to write data,
        * db = db, deg
        * ma = mag, deg
        * ri = real, imag
    \*args, \*\*kwargs :
        passed to `pandas.DataFrame.to_???`  functions.


    See Also
    --------
    networkset_2_spreadsheet : writes a spreadsheet for many networks
    """
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
    index = ntwk.frequency.f_scaled

    if form =='db':
        for m,n in ntwk.port_tuples:
            d[f'S{ntwk._fmt_trace_name(m,n)} Log Mag(dB)'] = \
                Series(ntwk.s_db[:,m,n], index = index)
            d[f'S{ntwk._fmt_trace_name(m,n)} Phase(deg)'] = \
                Series(ntwk.s_deg[:,m,n], index = index)
    elif form =='ma':
        for m,n in ntwk.port_tuples:
            d[f'S{ntwk._fmt_trace_name(m,n)} Mag(lin)'] = \
                Series(ntwk.s_mag[:,m,n], index = index)
            d[f'S{ntwk._fmt_trace_name(m,n)} Phase(deg)'] = \
                Series(ntwk.s_deg[:,m,n], index = index)
    elif form =='ri':
        for m,n in ntwk.port_tuples:
            d[f'S{ntwk._fmt_trace_name(m,n)} Real'] = \
                Series(ntwk.s_re[:,m,n], index = index)
            d[f'S{ntwk._fmt_trace_name(m,n)} Imag'] = \
                Series(ntwk.s_im[:,m,n], index = index)

    df = DataFrame(d)
    df.__getattribute__(f'to_{file_type}')(file_name,
        index_label=f'Freq({ntwk.frequency.unit})', **kwargs)

def network_2_dataframe(ntwk: Network, attrs: list[str] =None,
        ports: list[tuple[int, int]] = None, port_sep: str | None = None):
    """
    Convert one or more attributes of a network to a pandas DataFrame.

    Parameters
    ----------
    ntwk :  :class:`~skrf.network.Network` object
        the network to write
    attrs : list Network attributes
        like ['s_db','s_deg']
    ports : list of tuples
        list of port pairs to write. defaults to ntwk.port_tuples
        (like [(0,0)])
    port_sep : string
        defaults to None, which means a empty string "" is used for
        networks with lower than 10 ports. (s_db 11, s_db 21)
        For more than ten ports a "_" is used to avoid ambiguity.
        (s_db 1_1, s_db 2_1)

    Returns
    -------
    df : pandas DataFrame Object
    """
    if attrs is None:
        attrs = ["s_db"]
    if ports is None:
        ports = ntwk.port_tuples

    if port_sep is None:
        port_sep = "_" if ntwk.nports > 10 else ""

    d = {}
    for attr in attrs:
        attr_array = getattr(ntwk, attr)
        for m, n in ports:
            d[f'{attr} {m+1}{port_sep}{n+1}'] = attr_array[:, m, n]
    return DataFrame(d, index=ntwk.frequency.f)

def networkset_2_spreadsheet(ntwkset: NetworkSet, file_name: str = None, file_type: str = 'excel',
    *args, **kwargs):
    r"""
    Write a NetworkSet object to a spreadsheet, for your boss.

    Write  the s-parameters  of a each network in the networkset to a
    spreadsheet. If the `excel` file_type is used, then each network,
    is written to its own sheet, with the sheetname taken from the
    network `name` attribute.
    This functions makes use of the pandas module, which in turn makes
    use of the xlrd module. These are imported during this function.


    .. note::
        The frequency unit used in the spreadsheet is take from
        `ntwk.frequency.unit`


    Parameters
    ----------
    ntwkset :  :class:`~skrf.networkSet.NetworkSet` object
        the network to write
    file_name : str, None
        the file_name to write. if None,  ntwk.name is used.
    file_type : ['csv','excel','html']
        the type of file to write. See `pandas.DataFrame.to_???` functions.
    form : 'db','ma','ri'
        format to write data,
        * db = db, deg
        * ma = mag, deg
        * ri = real, imag
    \*args, \*\*kwargs :
        passed to `pandas.DataFrame.to_???` functions.


    See Also
    --------
    networkset_2_spreadsheet : writes a spreadsheet for many networks
    """
    if ntwkset.name is None and file_name is None:
        raise(ValueError('Either ntwkset must have name or give a file_name'))
    if file_name is None:
        file_name = ntwkset.name

    if file_type == 'excel':
        # add file extension if missing
        if not file_name.endswith('.xlsx'):
            file_name += '.xlsx'
        with ExcelWriter(file_name) as writer:
            [network_2_spreadsheet(k, writer, sheet_name=k.name, **kwargs) for k in ntwkset]
    else:
        [network_2_spreadsheet(k,*args, **kwargs) for k in ntwkset]


StringBuffer = StringIO


class TouchstoneEncoder(json.JSONEncoder):
    """
    Serializes Network object by converting arrays to lists,
    splitting complex numbers into real and imaginary,
    and breaking down frequency objects into dicts.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return np.real(obj), np.imag(obj)  # split into [real, im]
        if isinstance(obj, Frequency):
            return {'flist': obj.f_scaled.tolist(), 'funit': obj.unit}
        return json.JSONEncoder.default(self, obj)


def to_json_string(network):
    """
    Dumps Network to JSON string. Faster than converting and saving as touchstone.
    Safer than pickling (no arbitrary code execution on load).
    :param network: :class:`~skrf.network.Network` object
        A Network object to be serialized and returned as a JSON string.
    :return: str
        JSON string representation of a network object.
    """
    return json.dumps(network.__dict__, cls=TouchstoneEncoder)


def from_json_string(obj_string):
    """
    Loads network object from JSON string representation.
    :param obj_string: str
        JSON string representation of a network object.
    :return: :class:`~skrf.network.Network` object
        A Network object, rebuilt from JSON.
    """
    obj = json.loads(obj_string)
    ntwk = Network()
    ntwk.variables = obj['variables']
    ntwk.name = obj['name']
    ntwk.comments = obj['comments']
    ntwk.port_names = obj['port_names']
    ntwk.z0 = np.array(obj['_z0'])[..., 0] + np.array(obj['_z0'])[..., 1] * 1j  # recreate complex numbers
    ntwk.s = np.array(obj['_s'])[..., 0] + np.array(obj['_s'])[..., 1] * 1j
    ntwk.frequency = Frequency.from_f(np.array(obj['_frequency']['flist']),
                                         unit=obj['_frequency']['funit'])
    return ntwk
