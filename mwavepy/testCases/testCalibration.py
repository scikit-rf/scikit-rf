import unittest
import mwavepy as mv



class OnePortStandardCalibration(unittest.TestCase):
	'''
	One-port calibration test.

	loads data for a reciprocal embeding network, and some ideal
	standards. ficticous measurements are made by cascading the ideals
	behind teh embeding network, and a calibration is performed.

	the calculated embedding network and de-embeded ideals are compared
	to originals as a metric of 'working'
	
	'''
	def setUp(self):
		self.short = mv.Network('short.s1p')
		self.match = mv.Network('match.s1p')
		self.open = mv.Network('open.s1p')
		self.delay_short = mv.Network('delay short.s1p')
		self.embeding_network = mv.Network('embedingNetwork.s2p')
		

	def test_standard_calibration(self):
		ideals, measured = [],[]
		std_list = [self.short, self.match,self.open] 

		for ntwk in std_list:
			ideals.append(ntwk)
			measured.append(self.embeding_network**ntwk)

		cal = mv.Calibration(\
			ideals = ideals,\
			measured = measured,\
			type = 'one port',\
			frequency = self.short.frequency,\
			is_reciprocal = True,\
			)
		# did we find correct embeding network?
		self.assertEqual(self.embeding_network, cal.error_ntwk)
		# are the de-embeded networks the same as their ideals?
		for ntwk in std_list:
			self.assertEqual(ntwk,  cal.apply_cal(self.embeding_network**ntwk))

	def test_least_squares_calibration(self):
		ideals, measured = [],[]
		std_list = [self.short, self.match,self.open,self.delay_short] 

		for ntwk in std_list:
			ideals.append(ntwk)
			measured.append(self.embeding_network**ntwk)

		cal = mv.Calibration(\
			ideals = ideals,\
			measured = measured,\
			type = 'one port',\
			frequency = self.short.frequency,\
			is_reciprocal = True,\
			)
		# did we find correct embeding network?
		self.assertEqual(self.embeding_network, cal.error_ntwk)
		# are the de-embeded networks the same as their ideals?
		for ntwk in std_list:
			self.assertEqual(ntwk,  cal.apply_cal(self.embeding_network**ntwk))

	
suite = unittest.TestLoader().loadTestsFromTestCase(OnePortStandardCalibration)
unittest.TextTestRunner(verbosity=2).run(suite)

