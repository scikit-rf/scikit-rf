
#       io.py
#
#
#       Copyright 2012 alex arsenovic <arsenovic@virginia.edu>
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
.. module:: skrf.io.general
========================================
general (:mod:`skrf.io.general`)
========================================

General io functions for reading and writing skrf objects

.. autosummary::
    :toctree: generated/
    
    read
    read_all
    write
    write_all
    save_sesh
'''
import cPickle as pickle
from cPickle import UnpicklingError
import inspect
import os 
import zipfile
import warnings

from ..util import get_extn, get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import  Media
from ..networkSet import NetworkSet
from ..calibration.calibration import Calibration


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
    except(UnpicklingError):
        # if fid is seekable then reset to beginning of file
        fid.seek(0)
        
        if isinstance(file, basestring):
            # we created the fid so close it
            fid.close()
        raise
    
    if isinstance(file, basestring):
        # we created the fid so close it
        fid.close()
    
    return obj

def write(file, obj, overwrite = True):
    '''
    Write skrf object[s] to a file
    
    This uses the :mod:`pickle` module to write skrf objects to a file.
    Note that you can write any pickl-able python object. For example, 
    you can write  a list or dictionary of :class:`~skrf.network.Network`
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
    
    To make file written by this method cross-platform, the pickling 
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
    if isinstance(file, basestring):
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
        
        fid = open(file, 'wb')    
    
    else:
        fid = file
    
    pickle.dump(obj, fid, protocol=2)
    fid.close()
    
def read_all(dir='.', contains = None):
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
            
    return out
        
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
            file = open(os.path.join(dir+'/',filename),'w')
            write(file, obj,*args, **kwargs)
            file.close()
        except Exception as inst:
            print inst
            warnings.warn('couldnt write %s: %s'%(k,inst.strerror))
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
    print ('pickling: '),
    for k in dict_objs:
        try:
            if module  in inspect.getmodule(dict_objs[k]).__name__:
                try:
                    pickle.dumps(dict_objs[k])
                    if k[0] != '_':
                        objects[k] = dict_objs[k]
                        print k+', ',
                finally:
                    pass
               
        except(AttributeError, TypeError):
            pass
    if len (objects ) == 0:
        print 'nothing'
        
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
    ZVA-40, and possibly other network analyzers. It returns  into a 
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

    old_file = file(file_name,'r')
    new_file = open(new_file_name,'w')
    new_file.write('%s\n'%header_string)
    for line in old_file:
        new_file.write(line)
    new_file.close()
    old_file.close()

    if remove_tmp_file is True:
        os.rename(new_file_name,file_name)




