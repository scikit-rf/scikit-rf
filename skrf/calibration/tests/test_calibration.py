import unittest
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle
import skrf as rf
import numpy as npy
from numpy.random  import rand, uniform
from nose.tools import nottest
from nose.plugins.skip import SkipTest

from skrf.calibration import OnePort, PHN, SDDL, TRL, SOLT, UnknownThru, EightTerm, TwoPortOnePath, EnhancedResponse,TwelveTerm, SixteenTerm, LMR16, terminate, determine_line, determine_reflect

from skrf import two_port_reflect
from skrf.networkSet import NetworkSet

# number of frequency points to test calibration at 
# i choose 1 for speed, but given that many tests employ *random* 
# networks values >100 are better for  initialy verification
global NPTS  
NPTS = 1

WG =  rf.RectangularWaveguide(rf.F(75,100,NPTS), a=100*rf.mil,z0=50)



class DetermineTest(unittest.TestCase):
    def setUp(self):
        self.wg = WG
        wg = self.wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')  
        
        
        #thru
        self.T = wg.thru()
        self.T_m = self.embed(self.T)
        
        #line
        self.L = wg.line(80,'deg')
        self.L_approx = self.wg.line(90,'deg')
        self.L_m = self.embed(self.L)
        
        # reflect
        r = wg.load(-.8-.1j)
        self.R = two_port_reflect(r,r)
        self.R_approx = wg.short()
        self.R_m = self.embed(self.R)
        
        
    def embed(self,x):
        return self.X**x**self.Y    
        
    def test_determine_line(self):
        L_found = determine_line(self.T_m, self.L_m, 
                                 line_approx=self.L_approx)
        self.assertEqual(L_found,self.L)
    
    def test_determine_reflect(self):
        R_found = determine_reflect(self.T_m, self.R_m, self.L_m, 
                                 reflect_approx=self.R_approx)
        
        self.assertEqual(R_found,self.R.s11)
    
class CalibrationTest(object):
    '''
    This is the generic Calibration test case which all Calibration 
    Subclasses should be able to pass. They must implement
    '''
    def test_accuracy_of_dut_correction(self):
        a = self.wg.random(n_ports=self.n_ports, name = 'actual')
        m = self.measure(a)
        c = self.cal.apply_cal(m)
        c.name = 'corrected'   
        self.assertEqual(c,a)
        
    def test_error_ntwk(self):
        a= self.cal.error_ntwk 
    
    def test_coefs_ntwks(self):
        a= self.cal.coefs_ntwks
    
    def test_caled_ntwks(self):
        a= self.cal.caled_ntwks
        
    
    def test_residual_ntwks(self):
        a= self.cal.residual_ntwks
    
    def test_embed_then_apply_cal(self):
        
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.apply_cal(self.cal.embed(a)),a)
        
    def test_embed_equal_measure(self):
        
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.embed(a),self.measure(a))
        
    def test_from_coefs(self):
        cal_from_coefs = self.cal.from_coefs(self.cal.frequency, self.cal.coefs)
        ntwk = self.wg.random(n_ports=self.n_ports)
        if cal_from_coefs.apply_cal(self.cal.embed(ntwk))!= ntwk:
            raise ValueError
        self.assertEqual(cal_from_coefs.apply_cal(self.cal.embed(ntwk)),ntwk)
        
    def test_from_coefs_ntwks(self):
        cal_from_coefs = self.cal.from_coefs_ntwks(self.cal.coefs_ntwks)
        
        ntwk = self.wg.random(n_ports=self.n_ports)
        if cal_from_coefs.apply_cal(self.cal.embed(ntwk))!= ntwk:
            raise ValueError
        self.assertEqual(cal_from_coefs.apply_cal(self.cal.embed(ntwk)),ntwk)
        
class OnePortTest(unittest.TestCase, CalibrationTest):
    '''
    One-port calibration test.


    '''
    def setUp(self):
        self.n_ports = 1
        self.wg = WG
        wg = self.wg
                
        self.E = wg.random(n_ports =2, name = 'E')
        
        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.match( name='load'),
                ]
        measured = [self.measure(k) for k in ideals]
        
        self.cal = rf.OnePort(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
        
    def measure(self, ntwk):
        out = self.E**ntwk
        out.name = ntwk.name
        return out
    
    def test_accuracy_of_directivity(self):
        self.assertEqual(
            self.E.s11, 
            self.cal.coefs_ntwks['directivity'],
            )
        
    def test_accuracy_of_source_match(self):
        self.assertEqual(
            self.E.s22, 
            self.cal.coefs_ntwks['source match'],
            )
    
    def test_accuracy_of_reflection_tracking(self):
        self.assertEqual(
            self.E.s21*self.E.s12, 
            self.cal.coefs_ntwks['reflection tracking'],
            )
    
class SDDLTest(OnePortTest):
    def setUp(self):
        #raise SkipTest('Doesnt work yet')
        self.n_ports = 1
        self.wg = WG
        wg = self.wg
        
        self.E = wg.random(n_ports =2, name = 'E')
        #self.E.s[0,:,:] = npy.array([[.1j,1],[1j,1j+2]])
        #print self.E.s[0]
        
        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 33.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.SDDL(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
    
    def test_init_with_nones(self):
        wg=self.wg
        wg.frequency = rf.F.from_f([100])
        
        self.E = wg.random(n_ports =2, name = 'E')
        
        ideals = [
                wg.short( name='short'),
                None, 
                None,
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 33.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.SDDL(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
        self.cal.run()
    
    def test_from_coefs(self):
        raise SkipTest('not applicable ')
    def test_from_coefs_ntwks(self):
        raise SkipTest('not applicable ')

class SDDLWeikle(OnePortTest):
    def setUp(self):
        #raise SkipTest('Doesnt work yet')
        self.n_ports = 1
        self.wg = WG
        wg = self.wg
        self.E = wg.random(n_ports =2, name = 'E')
        #self.E.s[0,:,:] = npy.array([[.1j,1],[1j,1j+2]])
        #print self.E.s[0]
        
        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 80.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.SDDLWeikle(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
    
    def test_from_coefs(self):
        raise SkipTest('not applicable ')
    def test_from_coefs_ntwks(self):
        raise SkipTest('not applicable ')

class SDDMTest(OnePortTest):
    '''
    This is a specific test of SDDL to verify it works when the load is 
    a matched load. This test has been used to show that the SDDLWeikle 
    variant fails, with a perfect matched load. 
    '''
    def setUp(self):
        
        self.n_ports = 1
        self.wg = WG
        wg = self.wg
        
        self.E = wg.random(n_ports =2, name = 'E')
        
        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.match( name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 80.,'deg',name='qw'),
                wg.match(name='load'),
                ]
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.SDDL(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
    
    def test_from_coefs(self):
        raise SkipTest('not applicable ')
    
    def test_from_coefs_ntwks(self):
        raise SkipTest('not applicable ')

@SkipTest
class PHNTest(OnePortTest):
    '''
    '''
    def setUp(self):
        
        self.n_ports = 1
        self.wg = WG
        wg = self.wg
        
        self.E = wg.random(n_ports =2, name = 'E')
        known1 = wg.random()
        known2 = wg.random()
        #known1 = wg.short()
        #known2 = wg.load(rand() + rand()*1j) 
        
        ideals = [
                wg.delay_short( 45.,'deg',name='ideal ew'),
                wg.delay_short( 90.,'deg',name='ideal qw'),
                known1,
                known2,
                ]
        actuals = [
                wg.delay_short( 33.,'deg',name='true ew'),
                wg.delay_short( 110.,'deg',name='true qw'),
                known1,
                known2,
                ]
        measured = [self.measure(k) for k in actuals]
        self.actuals = actuals 
        
        self.cal = PHN(
            is_reciprocal = True, 
            ideals = ideals, 
            measured = measured,
            )
       
        
    def test_determine_ideals(self):
        self.cal.run()
        
        self.assertEqual(self.actuals[0], self.cal.ideals[0])
        self.assertEqual(self.actuals[1], self.cal.ideals[1])
            
    def test_from_coefs(self):
        raise SkipTest('not applicable')
    def test_from_coefs_ntwks(self):
        raise SkipTest('not applicable ')

class EightTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg =WG
        wg= self.wg
        
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
            
        measured = [self.measure(k) for k in ideals]
        
        self.cal = rf.EightTerm(
            ideals = ideals,
            measured = measured,
            switch_terms = (self.gamma_f, self.gamma_r)
            )
        
        
    def terminate(self, ntwk):
        '''
        terminate a measured network with the switch terms
        '''
        return terminate(ntwk,self.gamma_f, self.gamma_r)
        
    def measure(self,ntwk):
        out =  self.terminate(self.X**ntwk**self.Y)
        out.name = ntwk.name
        return out
    
    
    
    
        
    def test_unterminating(self):
        a = self.wg.random(n_ports=self.n_ports)
        #unermintated measurment
        ut =  self.X**a**self.Y
        #terminated measurement
        m = self.measure(a)
        self.assertEqual(self.cal.unterminate(m), ut)
        
       
    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.X.s11,
            self.cal.coefs_ntwks['forward directivity'])
    
    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.X.s22 , 
            self.cal.coefs_ntwks['forward source match'] )       
    
    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.X.s21 * self.X.s12 , 
            self.cal.coefs_ntwks['forward reflection tracking'])
    
    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Y.s11 , 
            self.cal.coefs_ntwks['reverse source match']   )     
    
    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Y.s22 , 
            self.cal.coefs_ntwks['reverse directivity']  )      
    
    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Y.s21 * self.Y.s12 , 
            self.cal.coefs_ntwks['reverse reflection tracking'])
    
    def test_k_accuracy(self):
        self.assertEqual(
            self.X.s21/self.Y.s12 , 
            self.cal.coefs_ntwks['k']  )   
    @nottest
    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)
            
class TRLTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg
        
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        #self.X.s[:,0,0] *=1e-1
        #self.Y.s[:,0,0] *=1e-1
        #self.X.s[:,1,1] *=1e-1 
        #self.Y.s[:,1,1] *=1e-1 
        
        actuals = [
            wg.thru( name='thru'),
            rf.two_port_reflect(wg.load(-.9-.1j),wg.load(-.9-.1j)),
            wg.attenuator(-3,True, 45,'deg')
            #wg.line(45,'deg',name='line'),
            ]
        self.actuals=actuals
        ideals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.line(90,'deg',name='line'),
            ]
            
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.TRL(
            ideals = ideals,
            measured = measured,
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    
    def test_found_line(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[2]==self.actuals[2])
        
    def test_found_reflect(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[1]==self.actuals[1])
            

class TRLWithNoIdealsTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg
        
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        #self.X.s[:,0,0] *=1e-1
        #self.Y.s[:,0,0] *=1e-1
        #self.X.s[:,1,1] *=1e-1 
        #self.Y.s[:,1,1] *=1e-1 
        
        ideals =  None
        
        actuals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.attenuator(-3,True, 45,'deg')
            ]
        self.actuals=actuals
        
        
        
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.TRL(
            ideals = ideals,
            measured = measured,
            switch_terms = (self.gamma_f, self.gamma_r)
            )
    
    
    def test_found_line(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[2]==self.actuals[2])
        
class TRLMultiline(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg
        
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        #self.X.s[:,0,0] *=1e-1
        #self.Y.s[:,0,0] *=1e-1
        #self.X.s[:,1,1] *=1e-1 
        #self.Y.s[:,1,1] *=1e-1 
        
        ideals =  None
        
        actuals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.short(nports=2, name='open'),
            wg.attenuator(-3,True, 45,'deg'),
            wg.attenuator(-6,True, 90,'deg'),
            wg.attenuator(-8,True, 145,'deg'),
            ]
        self.actuals=actuals
        
        
        
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.TRL(
            ideals = ideals,
            measured = measured,
            switch_terms = (self.gamma_f, self.gamma_r),
            n_reflects=2,
            )
    
    
    def test_found_line(self):
        self.cal.run()
        for k in range(2,5):
            self.assertTrue(self.cal.ideals[k]==self.actuals[k])         
        
class TREightTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        raise SkipTest()
        self.n_ports = 2
        self.wg = WG
        wg= self.wg
        
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
            
        measured = [self.measure_std(k) for k in ideals]
        
        cal1 = rf.TwoPortOnePath(
            ideals = ideals,
            measured = measured
            )
        switch_terms = (cal1.coefs_ntwks['forward switch term'],
                        cal1.coefs_ntwks['reverse switch term'])
        
        
        measured = [self.measure(k) for k in ideals]
        self.cal = rf.EightTerm(
            ideals = ideals,
            measured = measured,
            switch_terms = switch_terms, 
            )
        raise ValueError()
        
    def measure_std(self,ntwk):
        r= self.wg.random(2)
        m = ntwk.copy()
        mf = self.X**ntwk**self.Y
        
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = r.s[:,1,1]
        m.s[:,0,1] = r.s[:,0,1]
        return m    
    def measure(self,ntwk):
        
        m = ntwk.copy()
        mf = self.X**ntwk**self.Y
        mr = self.X**ntwk.flipped()**self.Y
        
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = mr.s[:,0,0]
        m.s[:,0,1] = mr.s[:,1,0]
        return m
        

        
class TwelveTermTest(unittest.TestCase, CalibrationTest):
    '''
    This test verifys the accuracy of the SOLT calibration. Generating 
    measured networks requires different error networks for forward and 
    reverse excitation states, these are described as follows
    
    forward excitation
        used for S21 and S11
        Mf = Xf ** S ** Yf  
    
    reverse excitation
        used for S12 and S22
        Mr = Xr ** S ** Yr
    
    
    '''
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
       
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.random(2,name='rand1'),
            wg.random(2,name='rand2'),
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        self.cal = rf.TwelveTerm(
            ideals = ideals,
            measured = measured,
            n_thrus=2, 
            )
    
    def measure(self,ntwk):
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        mr = self.Xr**ntwk**self.Yr
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,0,1] = mr.s[:,0,1]
        m.s[:,1,1] = mr.s[:,1,1]
        return m
        
    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.Xf.s11,
            self.cal.coefs_ntwks['forward directivity'])
    
    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.Xf.s22 , 
            self.cal.coefs_ntwks['forward source match'] )       
    
    def test_forward_load_match_accuracy(self):
        self.assertEqual(
            self.Yf.s11 , 
            self.cal.coefs_ntwks['forward load match'])
    
    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Xf.s21 * self.Xf.s12 , 
            self.cal.coefs_ntwks['forward reflection tracking'])
    
    def test_forward_transmission_tracking_accuracy(self):
        self.assertEqual(
            self.Xf.s21*self.Yf.s21 , 
            self.cal.coefs_ntwks['forward transmission tracking'])
    
    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Yr.s11 , 
            self.cal.coefs_ntwks['reverse source match']   )     
    
    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Yr.s22 , 
            self.cal.coefs_ntwks['reverse directivity']  )      
    
    def test_reverse_load_match_accuracy(self):
        self.assertEqual(
            self.Xr.s22 , 
            self.cal.coefs_ntwks['reverse load match'])
    
    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Yr.s21 * self.Yr.s12 , 
            self.cal.coefs_ntwks['reverse reflection tracking'])
    
    def test_reverse_transmission_tracking_accuracy(self):
        self.assertEqual(
            self.Yr.s12*self.Xr.s12 , 
            self.cal.coefs_ntwks['reverse transmission tracking'])
            
    
    @nottest
    def test_convert_12term_2_8term(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs))
        
        
        for k in converted:
            print(('{}-{}'.format(k,abs(self.cal.coefs[k] - converted[k]))))
        for k in converted:
            self.assertTrue(abs(self.cal.coefs[k] - converted[k])<1e-9)
        
    @nottest
    def test_convert_12term_2_8term_correction_accuracy(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs))
        
        self.cal._coefs = converted
        a = self.wg.random(n_ports=2)
        m = self.measure(a)
        c = self.cal.apply_cal(m)
               
        self.assertEqual(a,c)
    
    @nottest
    def test_verify_12term(self):
        
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)


class TwelveTermSloppyInitTest(TwelveTermTest):
    '''
    Test the TwelveTerm.__init__'s ability to 
    1) determine the number of thrus (n_thrus) hueristically
    2) put the standards in correct order if they use sloppy_input
    
    It must be a entirely seperate test because we want to ensure it 
    creates an accurate calibration.
    '''
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
       
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.attenuator(-20,name='atten'),
            wg.line(45,'deg',name = 'line') ,          
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        
        self.cal= TwelveTerm(
            ideals = NetworkSet(ideals).to_dict(), 
            measured = NetworkSet(measured).to_dict(),
            n_thrus=None,
            )
     
    def measure(self,ntwk):
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        mr = self.Xr**ntwk**self.Yr
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,0,1] = mr.s[:,0,1]
        m.s[:,1,1] = mr.s[:,1,1]
        return m    
    

class SOLTTest(TwelveTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
       
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            None,            
            ]
        actuals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(),            
            ]
    
        measured = [ self.measure(k) for k in actuals]
        
        self.cal = SOLT(
            ideals = ideals,
            measured = measured,
            n_thrus=1,
            )
        

class TwoPortOnePathTest(TwelveTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg =WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Yf = wg.random(n_ports =2, name='Yf')
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.random(2,name='rand1'),
            wg.random(2,name='rand2'),
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        self.cal = TwoPortOnePath(
            ideals = ideals,
            measured = measured,
            source_port=1,
            #n_thrus=2,
            )
    def measure(self,ntwk):
        r= self.wg.random(2)
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = r.s[:,1,1]
        m.s[:,0,1] = r.s[:,0,1]
        return m
        
    def test_accuracy_of_dut_correction(self):
        a = self.wg.random(n_ports=self.n_ports, name = 'actual')
        f = self.measure(a)
        r = self.measure(a.flipped())
        c = self.cal.apply_cal((f,r))
        c.name = 'corrected'   
        self.assertEqual(c,a)
        
    def test_embed_then_apply_cal(self):
        
        a = self.wg.random(n_ports=self.n_ports)
        f = self.cal.embed(a)
        r = self.cal.embed(a.flipped())
        self.assertEqual(self.cal.apply_cal((f,r)),a)
        
    def test_embed_equal_measure(self):
        # measurment procedure is different so this test doesnt apply
        raise SkipTest()
    
    def test_from_coefs(self):
        cal_from_coefs = self.cal.from_coefs(self.cal.frequency, self.cal.coefs)
        ntwk = self.wg.random(n_ports=self.n_ports)
    
    def test_from_coefs_ntwks(self):
        cal_from_coefs = self.cal.from_coefs_ntwks(self.cal.coefs_ntwks)
    def test_reverse_source_match_accuracy(self):
        raise SkipTest()   
    
    def test_reverse_directivity_accuracy(self):
        raise SkipTest()      
    
    def test_reverse_load_match_accuracy(self):
        raise SkipTest()  
    
    def test_reverse_reflection_tracking_accuracy(self):
        raise SkipTest()  
    
    def test_reverse_transmission_tracking_accuracy(self):
        raise SkipTest()  
    
    
    
    

class UnknownThruTest(EightTermTest):
    def setUp(self):
        
        self.n_ports = 2
        self.wg = WG
        wg= self.wg 
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        
        
        actuals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='match'),
            wg.impedance_mismatch(50,45)**wg.line(20,'deg',name='line')**wg.impedance_mismatch(45,50)

            ]
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='match'),
            wg.thru(name='thru'),
            ]
            
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.UnknownThru(
            ideals = ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r]
            )

class MRCTest(EightTermTest):
    def setUp(self):
        
        self.n_ports = 2
        self.wg = WG
        wg= self.wg 
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        
        
        def delay_shorts(d1,d2):
            ds1 = wg.delay_short(d1,'deg')
            ds2 = wg.delay_short(d2,'deg')
            return rf.two_port_reflect(ds1,ds2)
        
        actuals = [
            wg.short(nports=2, name='short'),
            delay_shorts(65,130),
            delay_shorts(120,75),
            wg.load(.2+.2j,nports=2, name='match'),
            wg.impedance_mismatch(50,45)**wg.line(20,'deg',name='line')**wg.impedance_mismatch(45,50)

            ]
        
        ideals = [
            wg.short(nports=2, name='short'),
            delay_shorts(45,90),
            delay_shorts(90,45),
            wg.load(.2+.2j,nports=2, name='match'),
            wg.thru(name='thru'),
            ]
            
        measured = [self.measure(k) for k in actuals]
        
        self.cal = rf.MRC(
            ideals = ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r]
            )
        
class TwelveTermToEightTermTest(unittest.TestCase, CalibrationTest):
    '''
    This test verifies the accuracy of the SOLT calibration, when used 
    on an error-box (8-term) model.
    
    
    '''
    def setUp(self):
        self.n_ports = 2
        wg= rf.wr10
        wg.frequency = rf.F.from_f([100])
        self.wg = wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        self.cal = rf.TwelveTerm(
            ideals = ideals,
            measured = measured,
            )
        
        coefs = rf.calibration.convert_12term_2_8term(self.cal.coefs, redundant_k=1)
        coefs = NetworkSet.from_s_dict(coefs,
                                    frequency=self.cal.frequency).to_dict()
        self.coefs= coefs
        
    def measure(self,ntwk):
        m = ntwk.copy()
        return terminate(m, self.gamma_f, self.gamma_r) 
    
    
    def test_forward_switch_term(self):
        self.assertEqual(self.coefs['forward switch term'], self.gamma_f)
    
    def test_forward_switch_term(self):
        self.assertEqual(self.coefs['reverse switch term'], self.gamma_r)
    @nottest
    def test_k(self):
        self.assertEqual(self.coefs['k'], self.X.s21/self.Y.s12 )
    
    
    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)

class LMR16Test(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        #Port 0: VNA port 0
        #Port 1: DUT port 0
        #Port 2: DUT port 1
        #Port 3: VNA port 1
        self.Z = wg.random(n_ports = 4, name = 'Z')

        r = wg.short(nports=1, name='short')
        m = wg.match(nports=1, name='load')
        mm = rf.two_port_reflect(m, m)
        rm = rf.two_port_reflect(r, m)
        mr = rf.two_port_reflect(m, r)
        rr = rf.two_port_reflect(r, r)

        thru_length = uniform(0,10)
        thru = wg.line(thru_length,'deg',name='line')

        self.thru = thru

        ideals = [
            thru,
            mm,
            rr,
            rm,
            mr
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.LMR16(
            measured = measured,
            ideals = [r],
            ideal_is_reflect = True,
             #Automatic sign detection doesn't work if the
             #error terms aren't symmetric enough
            sign = -1
            )

    def measure(self,ntwk):
        out = rf.connect(self.Z, 1, ntwk, 0, num=2)
        out.name = ntwk.name
        return out

    def test_solved_through(self):
        self.assertEqual(
            self.thru,
            self.cal.solved_through)

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s11,
            self.cal.coefs_ntwks['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s22 ,
            self.cal.coefs_ntwks['forward source match'] )

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s21 * self.Z.s12 ,
            self.cal.coefs_ntwks['forward reflection tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s33 ,
            self.cal.coefs_ntwks['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s44 ,
            self.cal.coefs_ntwks['reverse directivity']  )

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s34 * self.Z.s43 ,
            self.cal.coefs_ntwks['reverse reflection tracking'])

class SixteenTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        #Port 0: VNA port 0
        #Port 1: DUT port 0
        #Port 2: DUT port 1
        #Port 3: VNA port 1
        self.Z = wg.random(n_ports = 4, name = 'Z')

        o = wg.open(nports=1, name='open')
        s = wg.short(nports=1, name='short')
        m = wg.match(nports=1, name='load')
        om = rf.two_port_reflect(o, m)
        mo = rf.two_port_reflect(m, o)
        oo = rf.two_port_reflect(o, o)
        ss = rf.two_port_reflect(s, s)
        thru = wg.thru(name='thru')

        ideals = [
            thru,
            om,
            mo,
            oo,
            ss
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.SixteenTerm(
            measured = measured,
            ideals = ideals,
            )

    def measure(self,ntwk):
        out = rf.connect(self.Z, 1, ntwk, 0, num=2)
        out.name = ntwk.name
        return out

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s11,
            self.cal.coefs_ntwks['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s22 ,
            self.cal.coefs_ntwks['forward source match'] )

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s21 * self.Z.s12 ,
            self.cal.coefs_ntwks['forward reflection tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s33 ,
            self.cal.coefs_ntwks['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s44 ,
            self.cal.coefs_ntwks['reverse directivity']  )

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s34 * self.Z.s43 ,
            self.cal.coefs_ntwks['reverse reflection tracking'])


if __name__ == "__main__":
    unittest.main()
