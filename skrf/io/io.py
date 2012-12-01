
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


from ..convenience import get_extn, get_fid
from ..network import Network
from ..frequency import Frequency
from ..media import  Media
from ..networkSet import NetworkSet
from ..calibration.calibration import Calibration




global OBJ_EXTN # file extension conventions for skrf objects.
OBJ_EXTN = [
        [Frequency, 'freq'],
        [Network, 'ntwk'],
        [NetworkSet, 'ns'],
        [Calibration, 'cal'],
        [object, 'p'],
        ]


def read(file, *args, **kwargs):
    '''
    read  skrf object[s]
    
    reads a skrf object that is written with :func:`write` 
    
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
    '''
    fid = get_fid(file, mode='r')
    obj = pickle.load(fid, *args, **kwargs)
    fid.close()
    return obj

def write(file, obj, *args, **kwargs):
    '''
    Write skrf objects to disk using the pickle module
    
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

def save_skrfs(namespace, filename='skrfSesh.p'):
    skrf_objects = {}
    for k in namespace:
        try:
            if eval(k).__module__.split('.')[0] == 'skrf':
                if k[0]!='_':
                    print(k)
                    skrf_objects[k] = eval(k)
        except(AttributeError):
            pass
    write(skrf_objects, filename)
