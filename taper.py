from . network import cascade_list
from scipy  import linspace
from numpy import exp, log


class Taper1D(object):
    def __init__(self, med,  start, stop, n_sections, f,
                 length, length_unit='m', param='z0',f_is_normed=True, 
                 med_kw={}, f_kw={}):
        '''
        
        Parameters 
        ------------
        med : skrf.media.Media
            the media class, or a `@classmethod` `__init__`,  used to 
            generate the transmission line. see `med_kw` for arguments.
            examples: 
                * skrf.media.RectangularWaveguide # a class 
                * skrf.media.RectangularWaveguide.from_z0 # an init
            
        param : str
            name of the parameter of `med` that varies along the taper
        start : number
            starting value for `param`
        stop : number
            stop value for `param`
        n_sections : int
            number of sections in taper
        length : number
            physical length of the taper (in `length_unit`)
        length_unit : str 
            unit of length variable. see `skrf.to_meters`
            
        f : function
            function defining the taper transition. must take either 
            no arguments  or  take (x,length, start, stop). 
            see `f_is_normed` arguments
        f_is_normed: bool
            is `f` scalable and normalized. ie can f just be scaled
            to fit different start/stop conditions? if so then f is 
            called with no arguments, and must  have domain and raings
            of [0,1], and [0,1]
        
        f_kw : dict
            passed to `f()` when  called    
        
        med_kw : dict
            passed to `med.__init__` when an instance is created
        
        
        Notes
        -------
        the default behaviour should is to taper based on impedance. 
        to do this we inspect the `med` class for a `from_z0` 
        init method, and if it exists, we assign it to `med` attribute, 
        in `__init__`.
        addmitantly having `med` be a class or a method is abuse,
        it makes for a intuitive operation
    
        Examples
        ------------
        Create a linear taper from 100 to 1000mil
        
        
        >>> from skrf import Frequency, RectangularWaveguide, Taper1D, mil, inch
        >>> taper = Taper1D(med= RectangularWaveguide, 
                            param='a', 
                            start=100*mil, 
                            stop=1000*mil,
                            length=1*inch,
                            n_sections=20,
                            f=lambda x: x,
                            f_is_normed=True,
                            med_kw={'frequency':Frequency(75,110,101,'ghz')})
        '''
        self.med = med
        self.param = param
        self.start = start
        self.stop = stop
        self.f = f
        self.f_is_normed = f_is_normed
        self.length =length
        self.length_unit = length_unit
        self.n_sections= n_sections
        self.med_kw = med_kw
        self.f_kw = f_kw
        
        # the default behaviour should be to taper based on impedance. 
        # to do this we inspect the media class for a `from_z0` 
        # init method, and if it exists, we assign it to `med` attribute
        # admittedly having `med` be a class or a method is abuse,
        # it makes for a intuitive operation
        if param =='z0':
            if hasattr(self.med, 'from_z0'):
                self.med = getattr(self.med, 'from_z0')
    
    def __str__(self):
        return 'Taper: {classname}: {param} from {start}-{stop}'
        
    
    @property
    def section_length(self):
        return  1.0*self.length/self.n_sections
    
    @property
    def value_vector(self):
        if self.f_is_normed ==True:
            x = linspace(0,1,self.n_sections)
            y = self.f(x, **self.f_kw)*(self.stop-self.start) + self.start
        else:
            x = linspace(0,self.length,self.n_sections)
            y = self.f(x,self.length, self.start, self.stop, **self.f_kw)
        return y
    
    def media_at(self, val):
        '''
        creates a media instance for the taper with parameter value `val`
        '''
        med_kw = self.med_kw.copy() 
        med_kw.update({self.param:val})
        return self.med(**med_kw)
    
    def section_at(self,val):
        '''
        creates a single section of the taper with parameter value `val`
        '''
        return self.media_at(val).line(self.section_length,
                                       unit=self.length_unit)
    
    @property
    def medias(self):
        return [self.media_at(k) for k in self.value_vector]
    
    @property
    def sections(self):
        return [self.section_at(k) for k in self.value_vector]
    
    @property
    def ntwk(self):
        return cascade_list(self.sections)
    
    

class Linear(Taper1D):
    '''
    A linear Taper
    
    f(x)=x
    '''
    def __init__(self, **kw):
        opts = dict(f = lambda x:x, f_is_normed = True)
        kw.update(opts)
        super(Linear,self).__init__(**kw)


class Exponential(Taper1D):
    '''
    An exponential Taper
    
    f(x) = f0*e**(x/x1 * ln(f1/f0))
    
    where 
        f0: star param value
        f1: stop param value
        x: independent variable (position along taper)
        x1: length of taper 
    
    '''
    def __init__(self,**kw):
        
        def f(x,length, start, stop):
            return start*exp(x/length*(log(stop/start)))
        
        opts = dict(f = f, f_is_normed = False)
        kw.update(opts)
        super(Exponential,self).__init__(**kw)


class SmoothStep(Taper1D):
    '''
    A smoothstep Taper
    
    There is no analytical basis for this in the EE world that i know 
    of. it is just a reasonable smooth curve, that is easy to implement.
    
    f(x) = (3*x**2 - 2*x**3) 
    
    https://en.wikipedia.org/wiki/Smoothstep
    
    '''
    def __init__(self,**kw):
        
        f = lambda x:  3*x**2 - 2*x**3
        opts = dict(f=f, f_is_normed = True)
        kw.update(opts)
        super(SmoothStep,self).__init__(**kw)

