import unittest
import mwavepy as mv



		
class MediaTestCase(unittest.TestCase):
	'''
		
	'''
	def test_propagation_constant(self):
		self.media.propagation_constant
	
	def test_characterisitc_impedance_value(self):
		self.media.characteristic_impedance
		
	def test_match(self):
		self.media.match()
	
	def test_load(self):
		self.media.load(1)
	
	def test_short(self):
		self.media.short()
	
	def test_open(self):
		self.media.open()
	
	def test_capacitor(self):
		self.media.capacitor(1)
	
	def test_inductor(self):
		self.media.inductor(1)
	
	def test_impedance_mismatch(self):
		self.media.impedance_mismatch(1,2)
	
	def test_tee(self):
		self.media.tee()
	
	def test_splitter(self):
		self.media.splitter(4)
	
	
	

class FreespaceTestCase(MediaTestCase):
	def setUp(self):
		self.frequency = mv.Frequency(75,110,101,'ghz')
		self.media = mv.media.Freespace(self.frequency)
	
	def test_characterisitc_impedance_value(self):
		self.assertEqual(round(self.media.characteristic_impedance[0]) , 377)







suite = unittest.TestLoader().loadTestsFromTestCase(FreespaceTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
