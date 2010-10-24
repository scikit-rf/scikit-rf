import mwavepy as mv
from mwavepy.mathFunctions import dirac_delta
import pylab as plb
from numpy import zeros,pi,sinc,sin 
from scipy.constants import mil, micron 
## variational calculations
def junction_admittance(wg_I, wg_II, V_I, V_II, freq, M,N,\
	normalizing_mode=('te',1,0),**kwargs):
	'''
	calculates the junction admittance, and equivalent network for a
	discontinuity in a waveguide.

	takes:
		wg_I:
		wg_II:
		V_I:
		V_II:
		freq:
		M:
		N:
		normalzing mode:
		**kwargs:

	note:
		mode voltages are called like:
			V(mode_type,m,n,wg,**kwargs)
		so whatever you pass for the mode voltage functions must adhere
		to this convention
	'''
	

	## INPUTS
	# create vectors of mode-indecies and frequency
	m_ary = range(M)
	n_ary = range(N)
	f_ary = freq.f	
	F = freq.npoints

	
	# Calculate coupling coefficients. This is done this way, because the
	#	coupling is frequency independent
	V_II_mat = {'te':zeros((M,N)),'tm':zeros((M,N))}
	V_I_mat = {'te':zeros((M,N)),'tm':zeros((M,N))}
	for m in range(M):
		for n in range(N):
			for mode_type in ['te','tm']:
				V_I_mat[mode_type][m,n] =  V_I(mode_type,m,n,wg_I,**kwargs)
				V_II_mat[mode_type][m,n] =  V_II(mode_type,m,n,wg_II,**kwargs)
	
	
	# Calculate the admittances and store in array, the shape == FxMxN
	Y_II_mat = {'te': wg_II.y0('te', m_ary,n_ary, f_ary),\
		'tm':wg_II.y0('tm', m_ary,n_ary, f_ary)}

	Y_I_mat = {'te':wg_I.y0('te', m_ary,n_ary, f_ary),\
		'tm':wg_I.y0('tm', m_ary,n_ary, f_ary)}

	# calculate reaction matrix
	R_II_mat,R_I_mat = {},{}
	for mode_type in ['te','tm']:
		R_II_mat[mode_type] = V_II_mat[mode_type]**2 * Y_II_mat[mode_type] 			
		R_I_mat[mode_type] = V_I_mat[mode_type]**2 * Y_I_mat[mode_type] 			

	# remove normalizing mode frmo sum, but save it because it belongs
	# in teh denominator
	R_I_norm = R_I_mat[normalizing_mode[0]][:,normalizing_mode[1],normalizing_mode[2]].copy()
	R_I_mat[normalizing_mode[0]][:,normalizing_mode[1],normalizing_mode[2]]=0.0
	
	
	# sum total reaction
	R = R_II_mat['te'].sum(axis=1).sum(axis=1) +\
		R_II_mat['tm'].sum(axis=1).sum(axis=1) +\
		R_I_mat['te'].sum(axis=1).sum(axis=1) +\
		R_I_mat['tm'].sum(axis=1).sum(axis=1)

	
	# normalize to normalizing mode, usually the dominant mode
	Y_in_norm = R / R_I_norm

	#create a network type to return
	Gamma = (1-Y_in_norm)/(1+Y_in_norm)
	ntwk = mv.network.Network()
	ntwk.s = Gamma
	ntwk.frequency = freq

	output = {\
		'V_I_mat':	V_I_mat,\
		'V_II_mat':	V_II_mat,\
		'Y_I_mat':	Y_I_mat,\
		'Y_II_mat':	Y_II_mat,\
		'R_I_mat':	R_I_mat,\
		'R_II_mat':	R_II_mat,\
		'Y_in_norm':Y_in_norm,\
		'ntwk':	ntwk}
	return output

def junction_impedance(wg_I, wg_II, I_I, I_II, freq, M,N,\
	normalizing_mode=('te',1,0),**kwargs):

	## INPUTS
	# create vectors of mode-indecies and frequency
	m_ary = range(M)
	n_ary = range(N)
	f_ary = freq.f	
	F = freq.npoints

	
	# Calculate coupling coefficients. This is done this way, because the
	#	coupling is frequency independent
	I_II_mat = {'te':zeros((M,N)),'tm':zeros((M,N))}
	I_I_mat = {'te':zeros((M,N)),'tm':zeros((M,N))}
	for m in range(M):
		for n in range(N):
			for mode_type in ['te','tm']:
				I_I_mat[mode_type][m,n] =  I_I(mode_type,m,n,wg_I,**kwargs)
				I_II_mat[mode_type][m,n] =  I_II(mode_type,m,n,wg_II,**kwargs)
	
	
	# Calculate the admittances and store in array, the shape == FxMxN
	Z_II_mat = {'te': wg_II.z0('te', m_ary,n_ary, f_ary),\
		'tm':wg_II.z0('tm', m_ary,n_ary, f_ary)}

	Z_I_mat = {'te':wg_I.z0('te', m_ary,n_ary, f_ary),\
		'tm':wg_I.z0('tm', m_ary,n_ary, f_ary)}

	# calculate reaction matrix
	R_II_mat,R_I_mat = {},{}
	for mode_type in ['te','tm']:
		R_II_mat[mode_type] = I_II_mat[mode_type]**2 * Z_II_mat[mode_type] 			
		R_I_mat[mode_type] = I_I_mat[mode_type]**2 * Z_I_mat[mode_type] 			

	# remove normalizing mode frmo sum, but save it because it belongs
	# in teh denominator
	R_I_norm = R_I_mat[normalizing_mode[0]][:,normalizing_mode[1],normalizing_mode[2]].copy()
	R_I_mat[normalizing_mode[0]][:,normalizing_mode[1],normalizing_mode[2]]=0.0
	
	
	# sum total reaction
	R = R_II_mat['te'].sum(axis=1).sum(axis=1) +\
		R_II_mat['tm'].sum(axis=1).sum(axis=1) +\
		R_I_mat['te'].sum(axis=1).sum(axis=1) +\
		R_I_mat['tm'].sum(axis=1).sum(axis=1)

	
	# normalize to normalizing mode, usually the dominant mode
	Z_in_norm = R / R_I_norm

	#create a network type to return
	Gamma = (Z_in_norm-1)/(Z_in_norm+1)
	ntwk = mv.network.Network()
	ntwk.s = Gamma
	ntwk.frequency = freq

	output = {\
		'I_I_mat':	I_I_mat,\
		'I_II_mat':	I_II_mat,\
		'Z_I_mat':	Z_I_mat,\
		'Z_II_mat':	Z_II_mat,\
		'R_I_mat':	R_I_mat,\
		'R_II_mat':	R_II_mat,\
		'Z_in_norm':Z_in_norm,\
		'ntwk':	ntwk}
	return output



def converge_junction_admittance(y_tol = 1e-3, mode_rate=1, M_0=2,N_0=2,\
	min_converged=1, output=True, **kwargs):
	M,N=M_0,N_0
	y_old = junction_admittance(M=M, N=N,**kwargs)['ntwk'].y
	
	y_list = []
	converged_counter =0
	
	if output: print '(M,N)\t\tDelta_y'
	while converged_counter < min_converged:
		y_delta = 1
		while (y_delta > y_tol):
			if output:print '(%i,%i)\t\t%f'%(M,N, y_delta)
			
			M,N = int(M+mode_rate), int(N+mode_rate)
			out = junction_admittance(M=M, N=N,**kwargs)
			y_delta = abs(y_old-out['ntwk'].y).max()
			y_list.append(y_delta)
			y_old = out['ntwk'].y
		converged_counter +=1
		if output: print 'Converged #%i'%converged_counter
	out['convergence'] = y_list 
	return out


## mode voltages for specific discontinuties
def V_dominant_mode(mode_type,m,n,wg,**kwargs):
	'''
	mode voltage coupling of dominant mode to itself.

	assumed field is
		sin(pi*x/a) ; for 0 < x < wg.a, 0 < y < wg.b

	takes:
		mode_type: type of mode, a string, acceptable values are ['te','tmz']
		m: mode index in width dimension
		n: mode index in heigth dimension
		wg: a RectangularWaveguide type
	'''
	normalization = wg.eigenfunction_normalization(mode_type,m,n)
	coupling = wg.a*wg.b/2* dirac_delta(m-1)* dirac_delta(n)
	return normalization*coupling
	
def V_offset_dimension_change(mode_type,m,n,wg,x0,y0,a,b ):
	'''
	mode voltage coupling for dominant mode field of a rectangular
	aperture of dimension axb, radiating into a larger waveguide of
	dimensions wg.a x wg.b.

	assumed field is
		sin(pi/a*(x-x0)) ; for x0 < x < x0+a, y0 < y < y0+b

	takes:
		mode_type: type of mode, a string, acceptable values are ['te','tmz']
		m: mode index in width dimension
		n: mode index in heigth dimension
		wg: a RectangularWaveguide type
		x0: offset of aperture in width (a) dimension [m]
		y0: offset of aperture in height (b) dimension [m]
		a:	aperture size in width dimension
		b:	aperture size in height dimension
	returns:
		normalized  coupling value, a scalar. 
		
	'''
	A,B = wg.a,wg.b	
	#normalization_ap = 1/a#./A #  arbitrary, but  makes V_'s of reasonable units
	normalization = wg.eigenfunction_normalization(mode_type,m,n)
	a = a + 1e-12*a
	c = m*pi/A-pi/a
	d = m*pi/A+pi/a
	coupling = ((y0 + b)*sinc((y0 + b)*n/B) - y0*sinc(y0*n/B)) * 1/2.* (\
		(1/c*(sin(c*(x0+a)+x0*pi/a) - sin(c*(x0)+x0*pi/a) )) - \
		(1/d*(sin(d*(x0+a)-x0*pi/a) - sin(d*(x0)-x0*pi/a)) )) 
	return  normalization*coupling

## high level functions for discontinuity modeling
def translation_offset(wg, freq, delta_a, delta_b, **kwargs):
	'''
	calculates the response from a translation offset between two 
	rectangular waveguides. 
	
	takes:
		wg: RectangularWaveguide Object. 
		freq:	Frequency Object
		delta_a: offset in the width dimension [m]
		delta_b: offset in the height dimension [m]
		**kwargs: passed to converge_junction_admittance, see its help 
			for more info
	returns:
		ntwk: a Network type representing the junction
	'''
	wg_I = wg
	wg_II = wg
	
	junction_args = {\
		'a':abs(wg_I.a -delta_a),\
		'b': abs(wg_I.b -delta_b),\
		'x0': 0,\
		'y0': 0,\
		}
	kwargs.update(junction_args)
	out = converge_junction_admittance(\
		wg_I = wg_I,\
		wg_II = wg_II,\
		V_I = V_offset_dimension_change,\
		V_II = V_offset_dimension_change,\
		freq = freq,\
		**kwargs\
		)
	return out['ntwk']


def step_up(freq, wr_small, wr_big,  delta_a=0, delta_b=0, **kwargs):
	a = wr_small*10*mil
	b = a/2
	A = wr_big*10*mil +a/1e-6
	B = A/2
	x0,y0 = (A-a)/2, (B-b)/2 

	junction_args = {\
		'a':a,\
		'b': b,\
		'A': A,\
		'B': B,\
		'x0': x0,\
		'y0': y0,\
		}
	kwargs.update(junction_args)
	out = converge_junction_admittance(\
		wg_I = mv.RectangularWaveguide(wr_small*10*mil),\
		wg_II = mv.RectangularWaveguide(wr_big*10*mil, epsilon_R = 1-.001j),\
		V_I = dominant_mode,\
		V_II =offset_dimension_change,\
		freq = freq,\
		**kwargs\
		)
		
	return out['ntwk']



## OUTPUT
def show_coupling(out):
	for region in ['I','II']:
		for mode_type in ['te','tm']:
			figure()
			imshow((out['V_'+region+'_mat'][mode_type]**2).transpose(), \
				interpolation='nearest', origin = 'lower')
			title('Region %s: %s Modes'%(region, mode_type.upper()))
			xlabel('M-index')
			ylabel('N-index')
			colorbar()
	



def plot_mag_phase(out):
	figure()
	out['ntwk'].plot_s_deg()

	figure()
	out['ntwk'].plot_s_db()

	show();draw()
