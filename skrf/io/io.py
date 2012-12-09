
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
.. module:: skrf.io
========================================
io (:mod:`skrf.io.io`)
========================================

'''
import cPickle as pickle
from cPickle import UnpicklingError
import inspect
import os 
import zipfile

from ..helper import get_extn, get_fid
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
    Read  skrf object[s] from a file
    
    Reads a skrf object that is written with :func:`write`, which uses
    the pickle module.
    
    Parameters
    ------------
    file : str or file-object
        name of file, or  a file-object
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load
    
    Examples
    -------------
    >>> n = rf.Network('my_ntwk.s2p')
    >>> n.pickle('my_ntwk.ntwk')
    >>> n_unpickled = rf.read('my_ntwk.ntwk')
    
    See Also
    ------------
        :func:`network.Network.pickle`
        :func:`calibration.calibration.Calibration.pickle`
        
    Notes
    -------
    if file is a file-object it is left open, if it is a filename then 
    a file-object is opened and closed. If file is a file-object 
    and reading fails, then the position is reset back to 0 using seek.
    '''
    fid = get_fid(file, mode='r')
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

def write(file, obj, *args, **kwargs):
    '''
    Write skrf object[s] to a file
    
    This uses the pickle module to write skrf objects to a file.
    Note that you can write any python object, so, for example you can 
    write  a list or dict of Networks or Calibrations or anything. 
    
    Parameters
    ------------
    file : file or string 
        File or filename to which the data is saved.  If file is a 
        file-object, then the filename is unchanged.  If file is a 
        string, an appropriate extension will be appended to the file 
        name if it does not already have an extension.
    
    obj : an object, or list/dict of objects
        object or list/dict of objects to write to disk
    
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.dump
    
    Notes
    -------
    If no extension is provided with `file` one is chosen automatically. 
    Here are the extensions
    
    =====================  ===============
    skrf object            extension
    =====================  ===============
    :class:`Frequency`     '.freq'
    :class:`Network`       '.ntwk'
    :class:`NetworkSet`    '.ns'
    :class:`Calibration`   '.cal'
    :class:`Media`         '.med'
    other                  '.p'
    =====================  ===============  
    
    Examples
    -------------
    >>> n = rf.Network('my_ntwk.s2p')
    >>> rf.write('out',n)
    >>> n_red = rf.read('my_ntwk.ntwk')

    writing a mix of different objects
    
    >>> n = rf.Network('my_ntwk.s2p')
    >>> ns = rf.NetworkSet([n,n,n])
    >>> rf.write('out',[n,ns])
    >>> n_red = rf.read('my_ntwk.p')
    
    See Also
    ------------
    :func:`network.Network.pickle`
    :func:`calibration.calibration.Calibration.pickle`
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
        
        fid = open(file, 'w')    
    
    else:
        fid = file
    
    
    pickle.dump(obj, fid, *args, **kwargs)
    fid.close()
def read_all(dir='.', *args, **kwargs):
    out={}
    for filename in os.listdir(dir):
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
        
        

def write_all(locals, file='skrfSesh.p', module='skrf', exclude_prefix='_'):
    '''
    Writes all skrf objects in the current namespace.
    
    Can be used to save current workspace in a hurry. Note this can be 
    used for other modules as well by passing a different `module` name.
    
    Parameters
    ------------
    locals : dict
        the output of locals(). See the example. 
    file : str or file-object
        the file to save all objects to 
    module : str
        the module name to grep for. 
    exclude_prefix: str
        dont save objects which have this as a prefix. 
    
    
    Example
    ---------
    >>>rf.write_all(locals(), 'mysesh.p')
    
    '''
    objects = {}
    print ('pickling: '),
    for k in locals:
        try:
            if module  in inspect.getmodule(locals[k]).__name__:
                try:
                    pickle.dumps(locals[k])
                    if k[0] != '_':
                        objects[k] = locals[k]
                        print k+', ',
                finally:
                    pass
               
        except(AttributeError, TypeError):
            pass
    if len (objects ) == 0:
        print 'nothing'
        
    write(file, objects)




