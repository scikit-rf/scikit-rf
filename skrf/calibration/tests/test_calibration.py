import unittest
import os
import cPickle as pickle
import skrf as rf
import numpy as npy
from nose.tools import nottest



class CalibrationTest(object):
    '''
    This is the generic Calibration test case which all Calibration 
    Subclasses should be able to pass. They must implement
    '''
    def test_correction_accuracy_of_dut(self):
        a = self.wg.random(n_ports=self.n_ports)
        m = self.measure(a)
        c = self.cal.apply_cal(m)
        self.assertEqual(a,c)
        
    def test_error_ntwk(self):
        a= self.cal.error_ntwk 
    
    def test_coefs_ntwks(self):
        a= self.cal.coefs_ntwks
    
    def test_caled_ntwks(self):
        a= self.cal.caled_ntwks
    
    def test_residual_ntwks(self):
        a= self.cal.residual_ntwks
    
class OnePortTest(unittest.TestCase, CalibrationTest):
    '''
    One-port calibration test.


    '''
    def setUp(self):
        self.n_ports = 1
        self.wg = rf.wr10
        wg = self.wg
        wg.frequency = rf.F.from_f([100])
        
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
        return self.E**ntwk
    
    def test_directivity_accuracy(self):
        self.assertEqual(
            self.E.s11, 
            self.cal.coefs_ntwks['directivity'],
            )
        
    def test_source_match_accuracy(self):
        self.assertEqual(
            self.E.s22, 
            self.cal.coefs_ntwks['source match'],
            )
    
    def test_directivity_accuracy(self):
        self.assertEqual(
            self.E.s21*self.E.s12, 
            self.cal.coefs_ntwks['reflection tracking'],
            )
    
    
    '''
    def test_pickling(self):
        ideals, measured = [], []
        std_list = [self.short, self.match,self.open]

        for ntwk in std_list:
            ideals.append(ntwk)
            measured.append(self.embeding_network ** ntwk)

        cal = rf.OnePort(\
                ideals = ideals,\
                measured = measured,\
                type = 'one port',\
                is_reciprocal = True,\
                )
        
        original = cal
        
        f = open(os.path.join(self.test_dir, 'pickled_cal.cal'),'wb')
        pickle.dump(original,f)
        f.close()
        f = open(os.path.join(self.test_dir, 'pickled_cal.cal'))
        unpickled = pickle.load(f)
        a = unpickled.error_ntwk
        unpickled.run()
        
        
        # TODO: this test should be more extensive 
        self.assertEqual(original.ideals, unpickled.ideals)
        self.assertEqual(original.measured, unpickled.measured)
        
        f.close()
        os.remove(os.path.join(self.test_dir, 'pickled_cal.cal'))
    '''

class EightTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg= rf.wr10
        wg= self.wg
        wg.frequency = rf.F.from_f([100])
        
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
        m = ntwk.copy()
        ntwk_flip = ntwk.copy()
        ntwk_flip.flip()
        
        m.s[:,0,0] = (ntwk**self.gamma_f).s[:,0,0]
        m.s[:,1,1] = (ntwk_flip**self.gamma_r).s[:,0,0]
        m.s[:,1,0] = ntwk.s[:,1,0]/(1-ntwk.s[:,1,1]*self.gamma_f.s[:,0,0])
        m.s[:,0,1] = ntwk.s[:,0,1]/(1-ntwk.s[:,0,0]*self.gamma_r.s[:,0,0])
        return m
    
    
        
    def measure(self,ntwk):
        return self.terminate(self.X**ntwk**self.Y)
   
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
    
    @nottest
    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)
    
        
class TRLTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg= rf.wr10
        wg= self.wg
        #wg.frequency = rf.F.from_f([100])
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        self.X.s[:,0,0] *=1e-1
        self.Y.s[:,0,0] *=1e-1
        self.X.s[:,1,1] *=1e-1 
        self.Y.s[:,1,1] *=1e-1 
        
        actuals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.line(45,'deg',name='line'),
            ]
        
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
       
class SOLTTest(unittest.TestCase, CalibrationTest):
    '''
    This test verifys the accuracy of the SOLT calibration. Generating 
    measured networks requires different error networks for forward and 
    reverse excitation states, these are described as follows
    
    forward excition
        used for S21 and S11
        Mf = Xf ** S ** Yf  
    
    reverse excition
        used for S12 and S22
        Mr = Xr ** S ** Yr
    
    
    '''
    def setUp(self):
        self.n_ports = 2
        wg= rf.wr10
        wg.frequency = rf.F.from_f([100])
        self.wg = wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
       
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        self.cal = rf.SOLT(
            ideals = ideals,
            measured = measured,
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
            print('{}-{}'.format(k,abs(self.cal.coefs[k] - converted[k])))
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
    
    def test_verify_12term(self):
        import ipdb;ipdb.set_trace()
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)
        

class SOLTTest2(SOLTTest):
    '''
    This test verifys the accuracy of the SOLT calibration, when used 
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
        
        self.Xf = self.X.copy()
        self.Xr = self.X.copy()
        self.Yf = self.Y.copy()
        self.Yr = self.Y.copy()
        
        Y_term = self.terminate(self.Y)
        X_term = self.terminate(self.X)
        
        self.Yf.s[:,0,0] = Y_term.s[:,0,0]
        self.Yf.s[:,1,0] = Y_term.s[:,1,0]
        self.Xr.s[:,1,1] = X_term.s[:,1,1]
        self.Xr.s[:,0,1] = X_term.s[:,0,1]
        
        
        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]
        
    
        measured = [ self.measure(k) for k in ideals]
        
        self.cal = rf.SOLT(
            ideals = ideals,
            measured = measured,
            )
    def terminate(self, ntwk):
        '''
        terminate a measured network with the switch terms
        '''
        m = ntwk.copy()
        ntwk_flip = ntwk.copy()
        ntwk_flip.flip()
        
        m.s[:,0,0] = (ntwk**self.gamma_f).s[:,0,0]
        m.s[:,1,1] = (ntwk_flip**self.gamma_r).s[:,0,0]
        m.s[:,1,0] = ntwk.s[:,1,0]/(1-ntwk.s[:,1,1]*self.gamma_f.s[:,0,0])
        m.s[:,0,1] = ntwk.s[:,0,1]/(1-ntwk.s[:,0,0]*self.gamma_r.s[:,0,0])
        return m
        
    def measure(self,ntwk):
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        mr = self.Xr**ntwk**self.Yr
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,0,1] = mr.s[:,0,1]
        m.s[:,1,1] = mr.s[:,1,1]
        return m
    @nottest
    def test_12_2_8term(self):
        coefs = rf.calibration.convert_12term_2_8term(self.cal.coefs)
        coefs = rf.s_dict_to_ns(coefs, self.cal.frequency).to_dict()
        self.assertEqual(coefs['forward switch term'], self.gamma_f)
        self.assertEqual(coefs['reverse switch term'], self.gamma_r)
        self.assertEqual(coefs['k'], self.X.s21*self.Y.s21)
    
    
    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)
        
if __name__ == "__main__":
    unittest.main()
