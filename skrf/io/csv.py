
#       csv.py
#
#
#       Copyright 2013 alex arsenovic <arsenovic@virginia.edu>
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

from ..network import Network

def read_pna_csv(filename, *args, **kwargs):
    '''
    Reads data from a csv file written by an Agilient PNA
    
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
    fid = open(filename,'r')
    begin_line = -2
    end_line = -1
    comments = ''
    for k,line in enumerate(fid.readlines()):
        if line[0]=='!':
            comments+=line[1:]
        if line[:5] == 'BEGIN':
            begin_line = k
        elif line[:3] == 'END':
            end_line = k
        
        if k == begin_line+1:
            header = line
    
    fid.close()
    # TODO: use end_line to calculate skip_footer
    data = npy.genfromtxt(
        filename, 
        delimiter = ',',
        skip_header = begin_line+2,
        skip_footer = 1,
        *args, **kwargs
        )
    return header, comments, data 

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
    #TODO: check the data's format (Real-imag or db/angle , ..)
    header, comments, d = read_pna_csv(filename)
    
    return [ Network(
        f = d[:,0]*1e-9, 
        s = d[:,k*2+1]+1j*d[:,k*2+2],
        name = header.split('\"')[k*4+1],
        comments  = comments,
        ) \
        for k in range((d.shape[1]-1)/2) ]
        
