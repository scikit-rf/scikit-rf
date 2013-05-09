import unittest
import os
import cPickle as pickle
import skrf as rf
import numpy as npy

class OnePortTest(unittest.TestCase):
    '''
    One-port calibration test.


    '''
    def setUp(self):
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
            ideals = ideals, 
            measured = measured,
            )
        
        
    def measure(self, ntwk):
        return self.E**ntwk
    
    def test_correction_accuracy_of_standards(self):
        for k in range(self.cal.nstandards):
            self.assertEqual(self.cal.apply(self.cal.measured[k]),\
                self.cal.ideals[k])
    
    def test_correction_accuracy_of_dut(self):
        a = self.wg.random(n_ports=1)
        m = self.measure(a)
        c = self.cal.apply(m)
        self.assertEqual(a,c)
        
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
class EightTermTest(unittest.TestCase):
    def setUp(self):
        self.wg= rf.wr10
        wg= self.wg
        wg.frequency = rf.F.from_f([100])
        
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        
        
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
            )
    def measure(self,ntwk):
            return self.X**ntwk**self.Y
            
    def test_correction_accuracy_of_standards(self):
        for k in range(self.cal.nstandards):
            self.assertEqual(self.cal.apply(self.cal.measured[k]),\
                self.cal.ideals[k])
    
    def test_correction_accuracy_of_dut(self):
        a = self.wg.random(n_ports=2)
        m = self.measure(a)
        c = self.cal.apply(m)
        self.assertEqual(a,c)    
        
class SOLTTest(unittest.TestCase):
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
            
    def test_correction_accuracy_of_standards(self):
        for k in range(self.cal.nstandards):
            self.assertEqual(self.cal.apply(self.cal.measured[k]),\
                self.cal.ideals[k])
    
    def test_correction_accuracy_of_dut(self):
        
        a = self.wg.random(n_ports=2)
        m = self.measure(a)
        c = self.cal.apply(m)
        
        self.assertEqual(a,c)
    
    def test_convert_12term_2_8term(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs))
        
        #import pdb;pdb.set_trace()
        for k in converted:
            print('{}-{}'.format(k,abs(self.cal.coefs[k] - converted[k])))
        for k in converted:
            self.assertTrue(abs(self.cal.coefs[k] - converted[k])<1e-9)
        
    def test_convert_12term_2_8term_correction_accuracy(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs))
        
        #import pdb;pdb.set_trace()
        for k in converted:
            print('{}-{}'.format(k,abs(self.cal.coefs[k] - converted[k])))
        for k in converted:
            self.assertTrue(abs(self.cal.coefs[k] - converted[k])<1e-9)


