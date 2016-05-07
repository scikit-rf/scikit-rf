'''
.. module:: skrf.io.metas
========================================
metas (:mod:`skrf.io.metas`)
========================================

Functions for reading and writing file formats defined by METAS

http://www.metas.ch/metas/en/home/fabe/hochfrequenz/vna-tools.html

.. autosummary::
    :toctree: generated/

    
'''
from numpy import savetxt, array,hstack

from ..mathFunctions import complex2Scalar
from ..networkSet import NetworkSet
from ..network import average

def ns_2_sdatcv(ns,fname, polar=False):
    '''
    write a sdatcv from a skrf.NetworkSet
    '''
    
    ntwk = ns[0]
    nports = ntwk.nports
    nntwks = len(ns)
    nfreq = len(ntwk)
    freq_hz = ntwk.f.reshape(-1,1)
    
    
    
    ## make the header and columns information 
    # top junk
    top = '\n'.join(['SDATCV',
                   'Ports',
                   '\t'.join(['%i\t'%(k+1) for k in range(nports)])])
        
    #  port impedance info 
    z0ri = complex2Scalar(ntwk.z0[0])
    zcol='\t'.join(['Zr[%i]re\tZr[%i]im'%(k+1,k+1) \
        for k in range(nports)])
    zvals = '\t'.join([str(k) for k in z0ri])
    zhead='\n'.join([zcol,zvals])
    
    #  s and cov matrix info
    shead = '\t'.join(['S[%i,%i]%s'%(m+1, n+1,k) \
        for m,n in ntwk.port_tuples for k in ['re','im'] ])

    cvhead = '\t'.join(['CV[%i,%i]'%(n+1,m+1) \
        for m in range(2*nports**2) for n in range(2*nports**2)])

    datahead = '\t'.join(['Freq',shead,cvhead])

    header = '\n'.join([top,zhead,datahead])
    
    
    ## calculate covariance matrix
    cv = ns.cov()
    
    ## calculate mean s value,  everything so we have a 2D matrix
    mean_ntwk = average(ns, polar=polar)
    s_mean_flat =NetworkSet([mean_ntwk]).scalar_mat().squeeze()
    cv_flat = array([k.flatten('F') for k in cv])
    
    data = hstack([freq_hz,s_mean_flat, cv_flat])
    savetxt( fname,data,delimiter = '\t', header=header, comments='')




