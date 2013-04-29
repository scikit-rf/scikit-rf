
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


# general purpose objects 
class ObjectDict(collections.MutableMapping):
    def __init__(self, d,  attr_list=[], *args, **kwargs):
        self.store = dict(d,*args, **kwargs)
        
        [self.__setattr__(k,self.get_attr(k)) for k in attr_list]
    
    def __getitem__(self, key):
        return self.store[key]

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
        
    def search(self, **kwargs):
        for kw_key in kwargs:
            if kw_key[0] == '_':
                
                od = ObjectDict(
                    {k:self.store[k] for k in self.store if \
                    self.store[k].__getattribute__(kw_key[1:]) != kwargs[kw_key] }
                    )
            else:
                od = ObjectDict(
                    {k:self.store[k] for k in self.store if \
                    self.store[k].__getattribute__(kw_key) == kwargs[kw_key] }
                    )
            
        return od
    
    def get_attr(self, attr,  *args, **kwargs):
        '''
        get an attribute of each object in the dict. 
        
        if args is given, then the attribute is assumed to be a function and called with 
        *args and **kwargs
        '''
        if len(args) !=0 or len(kwargs)!=0:
            return ObjectDict({k:self.store[k].__getattribute__(attr)(*args, **kwargs) \
                for k in self.store \
                if self.store[k].__getattribute__(attr)(*args, **kwargs) is not None})
        else:    
            return ObjectDict({k:self.store[k].__getattribute__(attr) \
                for k in self.store \
                if self.store[k].__getattribute__(attr) is not None})


