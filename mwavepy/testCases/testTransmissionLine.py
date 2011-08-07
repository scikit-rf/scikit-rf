from numpy import imag, real,sign

import unittest
import mwavepy as mv



class FreeSpaceTestCase(unittest.TestCase):
	'''
	Test case for the FreeSpace class, which is the simplest 
	sub-class for GenericTEM.
	
	'''
	def setUp(self):
		'''
		create a vacum property, and test frequency 
		'''
		self.vacum = mv.transmissionLine.FreeSpace(1,1)	
		self.f = 1e9
	
	def test_sign_of_gamma(self):
		self.assertEqual(sign(imag(self.vacum.gamma(self.f))),1)
	def test_z0(self):
		self.assertEqual(round(real(self.vacum.z0(self.f))),377)	
			

suite = unittest.TestLoader().loadTestsFromTestCase(FreeSpaceTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
