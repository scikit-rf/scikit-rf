
#
#       util.py
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

.. currentmodule:: skrf.util
========================================
util (:mod:`skrf.util`)
========================================

Holds utility functions that are general conveniences.



General 
------------
.. autosummary::
   :toctree: generated/

   now_string
   find_nearest
   find_nearest_index
   get_fid
   get_extn



'''



import mathFunctions as mf

import warnings
import os
import cPickle as pickle
import pylab as plb
import numpy as npy
from scipy.constants import mil
from datetime import datetime
import collections, pprint
from subprocess import Popen,PIPE
# globals 


# other
def now_string():
    '''
    returns a unique sortable string, representing the current time
    
    nice for generating date-time stamps to be used in file-names, 
    the companion function :func:`now_string_2_dt` can be used 
    to read these string back into datetime objects.
    
    See Also
    ------------
    now_string_2_dt
    
    '''
    return datetime.now().__str__().replace('-','.').replace(':','.').replace(' ','.')

def now_string_2_dt(s):
    '''
    Converts the output of  :func:`now_string` to a datetime object.
    
    See Also
    -----------
    now_string
    
    '''
    return datetime(*[int(k) for k in s.split('.')])

def find_nearest(array,value):
    '''
    find nearest value in array.
    
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in 
    value : element of the array
        value to search for 
    
    Returns
    --------
    found_value : an element of the array 
        the value that is numerically closest to `value`
    
    '''
    idx=(npy.abs(array-value)).argmin()
    return array[idx]

def find_nearest_index(array,value):
    '''
    find nearest value in array.
    
    Parameters
    ----------
    array :  numpy.ndarray
        array we are searching for a value in 
    value : element of the array
        value to search for 
    
    Returns
    --------
    found_index : int 
        the index at which the  numerically closest element to `value`
        was found at
    
    
    taken from  http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    '''
    return (npy.abs(array-value)).argmin()

# file IO

def get_fid(file, *args, **kwargs):
    '''
    Returns a file object, given a filename or file object
    
    Useful  when you want to allow the arguments of a function to
    be either files or filenames
    
    Parameters
    -------------
    file : str or file-object
        file to open 
    \*args, \*\*kwargs : arguments and keyword arguments
        passed through to pickle.load
    '''
    if isinstance(file, basestring):
        return open(file, *args, **kwargs)
    else:
        return file
    
def get_extn(filename):
    '''
    Get the extension from a filename.
    
    The extension is defined as everything passed the last '.'.
    Returns None if it aint got one
    
    Parameters
    ------------
    filename : string 
        the filename 
    
    Returns
    --------
    ext : string, None
        either the extension (not including '.') or None if there 
        isnt one
        

    '''
    ext = os.path.splitext(filename)[-1]
    if len(ext)==0: 
        return None
    else:
        return ext[1:]

def basename_noext(filename):
    '''
    gets the basename and strips extension
    '''
    return os.path.splitext(os.path.basename(filename))[0]


# git
def git_version( modname):
    '''
    Returns output 'git describe', executed in a modules root directory.
    '''
    mod = __import__(modname)
    mod_dir =os.path.split(mod.__file__)[0] 
    p = Popen(['git', 'describe'], stdout = PIPE,stderr=PIPE, cwd =mod_dir )
    
    try:
        out,er = p.communicate()
    except(OSError):
        return None
        
    out = out.strip('\n')
    if out == '':
        return None
    return out
    


class ObjectList(object):
    def __init__(self, list_):
        self.list = list(list_)
    
    def __getattr__(self, name):
        return ObjectList([k.__getattribute__(name) for k in self.list])
    
    def __call__(self, *args, **kwargs):
        return ObjectList([k(*args, **kwargs) for k in self.store])


# general purpose objects 

class HomoList(collections.Sequence):
    '''
    A Homogeneous Sequence
    
    Provides a class for a list-like object which contains 
    homogeneous values. Attributes of the values can be accessed through
    the attributes of HomoList. Searching is done like numpy arrays.
    
    Initialized from a list  of all the same type
    
    >>> h = HomoDict([Foo(...), Foo(...)])
    
    The individual values of `h` can be access in identical fashion to 
    Lists.
    
    >>> h[0]
    
    Assuming that `Foo` has property `prop`  and function `func` ...
    
    Access elements' properties:
    
    >>> h.prop
    
    Access elements' functions:
        
    >>> h.func()
    
    Searching:
    
    >>> h[h.prop == value]
    >>> h[h.prop < value]
    
    Multiple search:
    
    >>> h[set(h.prop==value1) & set( h.prop2==value2)]
    
    Combos:
    
    >>> h[h.prop==value].func()
    '''

    
    def __init__(self, list_):
        self.store = list(list_)
        
    def __eq__(self, value):
        return [k for k in range(len(self)) if self.store[k] == value ]
    
    def __ne__(self, value):
        return [k for k in range(len(self)) if self.store[k] != value ]
    
    def __gt__(self, value):
        return [k for k in range(len(self)) if self.store[k] > value ]
    
    def __ge__(self, value):
        return [k for k in range(len(self)) if self.store[k] >= value ]
    
    def __lt__(self, value):
        return [k for k in range(len(self)) if self.store[k] < value ]
    
    def __le__(self, value):
        return [k for k in range(len(self)) if self.store[k] <= value ]
    
    def __getattr__(self, name):
        return self.__class__(
            [k.__getattribute__(name) for k in self.store])
        
    def __getitem__(self, idx):
        try: 
            return self.store[idx]
        except(TypeError):
            return self.__class__([self.store[k] for k in idx])
        
            
    def __call__(self, *args, **kwargs):
        return self.__class__(
            [k(*args,**kwargs) for k in self.store])
        
    def __setitem__(self, idx, value):
        self.store[idx] = value

    def __delitem__(self, idx):
        del self.store[idx]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        return pprint.pformat(self.store)
    
    def __repr__(self):
        return pprint.pformat(self.store)

class HomoDict(collections.MutableMapping):
    '''
    A Homogeneous Mutable Mapping
    
    Provides a class for a dictionary-like object which contains 
    homogeneous values. Attributes of the values can be accessed through
    the attributes of HomoDict. Searching is done like numpy arrays.
    
    Initialized from a dictionary containing values of all the same type
    
    >>> h = HomoDict({'a':Foo(...),'b': Foo(...), 'c':Foo(..)})
    
    The individual values of `h` can be access in identical fashion to 
    Dictionaries.
    
    >>> h['key']
    
    Assuming that `Foo` has property `prop`  and function `func` ...
    
    Access elements' properties:
    
    >>> h.prop
    
    Access elements' functions:
        
    >>> h.func()
    
    Searching:
    
    >>> h[h.prop == value]
    >>> h[h.prop < value]
    
    Multiple search:
    
    >>> h[set(h.prop==value1) & set( h.prop2==value2)]
    
    Combos:
    
    >>> h[h.prop==value].func()
    '''
    def __init__(self, dict_):
        self.store = dict(dict_)
        
    def __eq__(self, value):
        return [k for k in self.store if self.store[k] == value ]
    
    def __ne__(self, value):
        return [k for k in self.store if self.store[k] != value ]
    
    def __gt__(self, value):
        return [k for k in self.store if self.store[k] > value ]
    
    def __ge__(self, value):
        return [k for k in self.store if self.store[k] >= value ]
    
    def __lt__(self, value):
        return [k for k in self.store if self.store[k] < value ]
    
    def __le__(self, value):
        return [k for k in self.store if self.store[k] <= value ]
    
    def __getattr__(self, name):
        return self.__class__(
            {k: self.store[k].__getattribute__(name) for k in self.store})
        
    def __getitem__(self, key):
        c =   self.__class__({k:self.store[k] for k in key})
        if len(c) == 1: 
            return c.store.values()[0]
        else: 
            return c
            
    def __call__(self, *args, **kwargs):
        return self.__class__(
            {k: self.store[k](*args, **kwargs) for k in self.store})
        
    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        return pprint.pformat(self.store)
    
    def __repr__(self):
        return pprint.pformat(self.store)
        
    
    
