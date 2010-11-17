import unittest
import mwavepy as mv

class NetworkOperationsTestCase(unittest.TestCase):
	'''
	Network class operation test case. 
	
	 As tested by lihan in ADS the following is true:
		test 3 == test1 ** test2
	
	'''
	def setUp(self):
		self.ntwk1 =mv.Network('ntwk1.s2p')
		self.ntwk2 =mv.Network('ntwk2.s2p')
		self.ntwk3 =mv.Network('ntwk3.s2p')

	def test_cascade(self):
		self.assertEqual(self.ntwk1**self.ntwk2, self.ntwk3)

	def test_de_embed_by_inv(self):
		self.assertEqual(self.ntwk1.inv**self.ntwk3,self.ntwk2)
		self.assertEqual(self.ntwk3**self.ntwk2.inv,self.ntwk1)

if __name__ == "__main__":
   unittest.main()
