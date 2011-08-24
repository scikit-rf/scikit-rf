import unittest
import mwavepy as mv



		
class TransmissionLineTestCase(unittest.TestCase):
	'''
	
	
	'''
	def test_freespace_simple(self):
		tline = mv.transmissionLine.FreeSpace()
	
	def test_recangular_waveguide(self):
		tline = mv.transmissionLine.RectangularWaveguide(10)	
	
	def test_generic_tem(self):
		tline = mv.transmissionLine.GenericTEM(1,1,1,1)
	

		

suite = unittest.TestLoader().loadTestsFromTestCase(TransmissionLineTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
