from . network import cascade_list
from scipy  import linspace

class Taper1D(object):
    def __init__(self, med, param, start, stop, n_sections, length,f,kw={}):
        '''
        
        Parameters 
        ------------
        med : skrf.media.Media
            the class used to generate the transmission line
        param : str
            name of the parameter of `med` that varies along the taper
        start : number
            starting value for `param`
        stop : number
            stop value for `param`
        n_sections : int
            number of sections in taper
        length : number
            physical length of the taper (in meters)
        f : function
            function defining the taper transition.  domain and range 
            should both be between (0,1)
        kw : dict
            passed to `med.__init__` when an instance is created
        
        
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
                            kw={'frequency':Frequency(75,110,101,'ghz')})
        '''
        self.med = med
        self.param = param
        self.start = start
        self.stop = stop
        self.f = f
        self.length =length
        self.n_sections= n_sections
        self.kw = kw

    
    @property
    def section_length(self):
        return  self.length/self.n_sections
    
    @property
    def value_vector(self):
        x = linspace(0,1,self.n_sections)
        return self.f(x)*(self.stop-self.start) + self.start
    
    def media_at(self, val):
        '''
        creates a media instance for the taper with parameter value `val`
        '''
        kw = self.kw.copy() 
        kw.update({self.param:val})
        return self.med(**kw)
    
    def section_at(self,val):
        '''
        creates a single section of the taper with parameter value `val`
        '''
        return self.media_at(val).line(self.section_length,unit='m')
    
    @property
    def sections(self):
        return [self.section_at(k) for k in self.value_vector]
    
    @property
    def ntwk(self):
        return cascade_list(self.sections)
    
    


