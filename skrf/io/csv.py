
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
import os
from ..network import Network
from .. import mathFunctions as mf

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
    from pandas import Series, Index, DataFrame
    header, comments, d = read_pna_csv(filename)
    f_unit = header.split(',')[0].split(')')[0].split('(')[1]
    
    names = header.split(',')
    
    index = Index(d[:,0], name = names[0])
    df=DataFrame({names[k]:d[:,k] for k in range(1,len(names))}, index=index)
    return df
    
def pna_csv_2_ntwks2(filename, *args, **kwargs):    
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
    
    try:
        names = [header.split('\"')[k*4+1] for k in range((d.shape[1]-1)/2) ]
    except(IndexError):
        try:
            names = [header.split(',')[k*2+1] for k in range((d.shape[1]-1)/2) ]
        except(IndexError):
            names = [os.path.basename(filename).split('.')[-2]+str(k) \
                for k in range(d.shape[1]-1)/2 ]
    
    ntwk_list = []
    
    
    
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
        

