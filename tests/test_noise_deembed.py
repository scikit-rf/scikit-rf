import numpy as np
import skrf as rf
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.tri as tri
from scipy.interpolate import interp1d

#%%
def plot_contour(x,y,z,cmap, **kw) :
    ri =  np.linspace(0,1, 50); 
    ti =  np.linspace(0,2*np.pi, 150);
    Ri , Ti = np.meshgrid(ri, ti)
    xi = np.linspace(-1,1, 50);    
    Xi, Yi = np.meshgrid(xi, xi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Zi = interpolator(Xi, Yi)
    fig, ax = plt.subplots()
    an = np.linspace(0, 2*np.pi, 50)
    cs,sn=np.cos(an), np.sin(an)
    plt.plot(cs,sn, color='k', lw=0.25)
    plt.plot(cs,sn*0, color='g', lw=0.25)
    plt.plot((1+cs)/2, sn/2, color='k', lw=0.25)
    plt.axis('equal')
    ax.set_axis_off()
    ax.contour(Xi, Yi, Zi, levels=20, vmin=Zi.min(), vmax= Zi.max(), linewidths=0.5,  colors='k')
    cntr1 = ax.contourf(Xi, Yi, Zi, levels=20, vmin=Zi.min(), vmax= Zi.max(),cmap=cmap, **kw)
    fig.colorbar(cntr1, ax=ax)
    ax.plot(x, y, 'o', ms=0.3, color='k')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.show()
    if None:
        plot_contour(x,y,z,nm,cmap, nfmin_CPtun['XY'])
        nws.s.s11.plot_s_smith( marker='o', ms=3) 
#%%


if True :
    def_fixture_s2p =   'Port1_path.s2p'                                         
    thru_s2p = 'thru.s2p'
    WG_MS = rf.Network(def_fixture_s2p, name = 'WG')['75-75.05GHz']
    thru = rf.Network(thru_s2p, name = 'WG')['75-75.05GHz']
    
    Ncascade = WG_MS ** thru

    print (Ncascade.nfmin_db, Ncascade.g_opt, Ncascade.rn)

    deembed = WG_MS.inv
    deembed ** Ncascade
    
    orig =  deembed ** Ncascade
    
    
    newnetw = thru.copy()
    newnetw.set_noise_a(thru.noise_freq, nfmin_db=4.5, gamma_opt=complex(.7,-0.2), rn=10 )
    newnetw_nfmin = newnetw.nfdb_gs(complex(.7,-0.2))
    print (str(complex(.7,-0.2)),'=>', newnetw_nfmin)

    r = np.linspace(0.1,0.9,9)
    a = np.linspace(0,2*np.pi,21)
    r_, a_ = np.meshgrid(r,a)
    c_ = r_ *np.exp(1j * a_)
    g= c_.flatten()
    x =  np.real(g)
    y =  np.imag(g)
    z = newnetw.nfdb_gs(g)
    z[:,0].shape
    x.shape
    y.shape
    plot_contour(x,y,z[:,0],cm.plasma_r) 
    
    print (str(complex(.7,-0.2)),'=>', newnetw_nfmin)
    

    print (orig.nfmin_db, orig.g_opt, orig.rn)
    print (thru.nfmin_db, thru.g_opt, thru.rn)
    print (newnetw.nfmin_db, newnetw.g_opt, newnetw.rn)


