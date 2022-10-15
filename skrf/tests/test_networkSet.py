import unittest
import os
import numpy as np
import skrf as rf
import glob

class NetworkSetTestCase(unittest.TestCase):
    """
    NetworkSet class operation test case.
    """

    def setUp(self):
        """
        Initialize tests.
        """
        # Touchstone files
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.ntwk4 = rf.Network(os.path.join(self.test_dir, 'ntwk4.s2p'))

        # dummy networks of various frequency and port shapes
        self.freq1 = rf.Frequency(75, 110, 101, 'ghz')
        self.freq2 = rf.Frequency(75, 110, 201, 'ghz')
        self.ntwk_freq1_1p = rf.Network(frequency=self.freq1)
        self.ntwk_freq1_1p.s = np.random.rand(len(self.freq1), 1, 1)
        self.ntwk_freq1_2p = rf.Network(frequency=self.freq1)
        self.ntwk_freq1_2p.s = np.random.rand(len(self.freq1), 2, 2)
        self.ntwk_freq2_1p = rf.Network(frequency=self.freq2)
        self.ntwk_freq2_1p.s = np.random.rand(len(self.freq2), 1, 1)
        self.ntwk_freq2_2p = rf.Network(frequency=self.freq2)
        self.ntwk_freq2_2p.s = np.random.rand(len(self.freq2), 2, 2)

        # dummy networks with associated parameters
        # total number of different networks       
        self.params = [
                {'a':0, 'X':10, 'c':'A'},
                {'a':1, 'X':10, 'c':'A'},
                {'a':2, 'X':10, 'c':'A'},
                {'a':1, 'X':20, 'c':'A'},
                {'a':0, 'X':20, 'c':'A'},
            ]
        # for write_mdif
        self.params_datatypes = {'a': 'int', 'X': 'double', 'c': 'string'}
        
        # create M dummy networks
        self.ntwks_params = [rf.Network(frequency=self.freq1, 
                                        s=np.random.rand(len(self.freq1),2,2), 
                                        name=f'ntwk_{m}',
                                        comment=f'ntwk_{m}',
                                        params=params) \
                             for (m, params) in enumerate(self.params) ]
        
        # Test nominal
        self.ns = rf.NetworkSet([self.ntwk1, self.ntwk2, self.ntwk3])

        # Create NetworkSet from a list of Network containing a .params dict parameters
        self.ns_params = rf.NetworkSet(self.ntwks_params)        

    def test_constructor(self):
        """
        Test the `NetworkSet()` constructor.
        """
        # NetworkSet without input parameter is an empty NetworkSet
        self.assertEqual(rf.NetworkSet(), rf.NetworkSet([]))

        # the required parameter must be a list
        self.assertRaises(ValueError, rf.NetworkSet, 0)
        self.assertRaises(ValueError, rf.NetworkSet, 'wrong')
        self.assertRaises(ValueError, rf.NetworkSet, False)

        # all elements should be of Network type
        self.assertRaises(TypeError, rf.NetworkSet, [self.ntwk1, 0])
        self.assertRaises(TypeError, rf.NetworkSet, [self.ntwk1, 'wrong'])

        # all Networks should share the same Frequency
        self.assertRaises(ValueError, rf.NetworkSet, [self.ntwk_freq1_1p, self.ntwk_freq2_1p])

        # all Networks should share the same number of ports
        self.assertRaises(ValueError, rf.NetworkSet, [self.ntwk_freq1_1p, self.ntwk_freq1_2p])

        # expected situations: same number of ports and frequencies
        ntwk_set1 = rf.NetworkSet([self.ntwk_freq1_1p, self.ntwk_freq1_1p])
        ntwk_set2 = rf.NetworkSet([self.ntwk_freq2_1p, self.ntwk_freq2_1p])

    def test_from_zip(self):
        """
        Test the `NetworkSet.from_zip()` constructor class method.
        """
        # reading a zip of touchstone
        zip_filename = os.path.join(self.test_dir, 'ntwks.zip')
        ntwk_set = rf.NetworkSet.from_zip(zip_filename)
        # reading a zip of pickled networks
        zip_filename2 = os.path.join(self.test_dir, 'ntwk_pickle.zip')
        ntwk_set = rf.NetworkSet.from_zip(zip_filename2)

    def test_from_dir(self):
        """
        Test the `NetworkSet.from_dir()` constructor class method.
        """
        dir_path = os.path.join(self.test_dir, './ntwks')
        ntwk_set = rf.NetworkSet.from_dir(dir_path)

    def test_from_s_dict(self):
        """
        Test the `NetworkSet.from_s_dict()` constructor class method.
        """
        d = {'ntwk1': self.ntwk1.s,
             'ntwk2': self.ntwk2.s, 
             'ntwk3': self.ntwk3.s}
        ntwk_set = rf.NetworkSet.from_s_dict(d, frequency=self.ntwk1.frequency)

    def test_to_dict(self):
        """
        Test the `to_dict()` method.
        """
        d = self.ns.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(len(d), len(self.ns))
        for (key, ntwk) in zip(d.keys(), self.ns):
            self.assertEqual(d[key], ntwk)
        
    def test_to_s_dict(self):
        """
        Test the `to_s_dict()` method.
        """
        s_d = self.ns.to_s_dict()
        self.assertIsInstance(s_d, dict)
        self.assertEqual(len(s_d), len(self.ns))
        for (key, ntwk) in zip(s_d.keys(), self.ns):
            np.testing.assert_equal(s_d[key], ntwk.s)

    def test_copy(self):
        """
        Test the `copy()`method and the equality of two NetworkSets
        """
        copy = self.ns.copy()
        self.assertEqual(copy, self.ns)
        
        copy_inversed = rf.NetworkSet([self.ntwk3, self.ntwk2, self.ntwk1])
        self.assertNotEqual(copy_inversed, self.ns)


    def test_sort(self):
        """    
        Test the `sort` method.
        """
        ns_unsorted = rf.NetworkSet([self.ntwk2, self.ntwk1, self.ntwk3])
        # not inplace sorting
        ns_sorted = ns_unsorted.sort(inplace=False)
        for (idx, ntwk) in enumerate(ns_sorted):
            self.assertEqual(ntwk.name, f'ntwk{idx+1}')
            
        # inplace sorting
        ns_unsorted.sort()
        for (idx, ntwk) in enumerate(ns_unsorted):
            self.assertEqual(ntwk.name, f'ntwk{idx+1}')  
            
        # sorting with respect to a property
        self.ntwk1.dummy = 100
        self.ntwk2.dummy = 10
        self.ntwk3.dummy = 40
        ns_unsorted = rf.NetworkSet([self.ntwk2, self.ntwk1, self.ntwk3])
        ns_unsorted.sort(key=lambda x: x.dummy)  # dummy -> 10, 40, 100
        self.assertEqual(ns_unsorted, 
                         rf.NetworkSet([self.ntwk2, self.ntwk3, self.ntwk1]))
        
    def test_filter(self):
        """
        Test the `filter` method.
        """
        ns_unfiltered = rf.NetworkSet([self.ntwk2, self.ntwk1, self.ntwk3])
        ns_filtered = ns_unfiltered.filter('ntwk2')
        self.assertEqual(len(ns_filtered), 1)
        self.assertEqual(ns_filtered[0], self.ntwk2)

    def test_scalar_mat(self):
        """
        Test the `scalar_mat` method.
        """
        mat = self.ns.scalar_mat()
        # check the resulting shape
        self.assertEqual(mat.shape, (len(self.ns[0].f), len(self.ns), 2*self.ns[0].nports**2))

    def test_inv(self):
        """
        Test the `inv` method.
        """
        ns_inverted = self.ns.inv
        for (ntwk, ntwk_inv) in zip(self.ns, ns_inverted):
            self.assertEqual(ntwk.inv, ntwk_inv)
            
    def test_write(self):
        """
        Test the `write` method.
        """
        ns = self.ns.copy()
        # should fail if no name is given to the NetworkSet or not filename
        self.assertRaises(ValueError, ns.write)
        ns.write(file='testing.ns')
        
        # using a name        
        ns.name = 'testing'
        ns.write()  # write 'testing.ns'
        os.remove('testing.ns')
        
    def test_write_spreadsheet(self):
        """
        Test the `write_spreadsheet` method.
        """
        ns = self.ns.copy()
        # should fail if not NetworkSet.name attribute exist
        self.assertRaises(ValueError, ns.write_spreadsheet)
        # passing a name
        ns.name = 'testing'
        ns.write_spreadsheet()  # write 'testing.xlsx'
        os.remove('testing.xlsx')
        # passing a filename
        ns.write_spreadsheet(file_name='testing2.xlsx')
        os.remove('testing2.xlsx')
        
    def test_ntwk_attr_2_df(self):
        """
        Test the `ntwk_attr_2_df` method.
        """
        df = self.ns.ntwk_attr_2_df('s_db', m=1, n=0)

    def test_interpolate_from_network(self):
        """
        Test the `interpolate_from_network` method.

        """
        param = [1, 2, 3]
        x0 = 1.5
        interp_ntwk = self.ns.interpolate_from_network(param, x0)

    def test_has_params(self):
        """ Test the .has_params() method """
        # all params have been set
        self.assertTrue(self.ns_params.has_params())

        # at least one params property is None (>=v0.21.0)
        self.assertFalse(self.ns.has_params())
        
        # at least one params property does not exist (<v0.21.0)
        ns_wo_params = self.ns.copy()
        for ntwk in ns_wo_params:
            delattr(ntwk, 'params')
        self.assertFalse(ns_wo_params.has_params())
               
        # some params do not have the same length
        ns_wo_all_same_params_length = self.ns_params.copy()
        ns_wo_all_same_params_length[0].params = {'a': 0, 'c': 'A'}
        self.assertFalse(ns_wo_params.has_params())
        ns_wo_all_same_params_length[0].params = {'a': 0, 'b': 10, 'c':'A', 'd':'hu ho'}
        self.assertFalse(ns_wo_params.has_params())
        
        # not all keys of the params are the same
        ns_params_diff_keys = self.ns_params.copy()
        ns_params_diff_keys[0].params = {'X': 0, 'Y': 10, 'Z' : 'A'}
        self.assertFalse(ns_params_diff_keys.has_params())        

    def test_dims_param(self):
        """ Tests associated to the .dims parameter """
        from collections import Counter
        
        # unassigned params NetworkSet
        self.assertEqual(Counter(self.ns.dims), Counter(None))
        # assigned params
        expected_dims = self.params[0].keys()
        self.assertEqual(Counter(self.ns_params.dims), Counter(expected_dims))

    def test_coords_param(self):
        """ Tests associated to the .coords parameter """
        from collections import Counter
        
        # unassigned params NetworkSet
        self.assertEqual(Counter(self.ns.coords), Counter(None))

        # assigned params
        # get a dict of unique values for each param
        expected_coords = {p: [] for p in self.params[0]}
        for params in self.params:
            for p in expected_coords.keys():
                expected_coords[p].append(params[p])
        for p in expected_coords.keys():
            expected_coords[p] = list(set(expected_coords[p]))

        self.assertEqual(Counter(self.ns_params.coords), Counter(expected_coords))

    def test_params_param(self):
        """ Test the params property """
        self.assertEqual(self.ns_params.params, list(self.ns_params.dims))
        self.assertEqual(self.ns.params, [])

    def test_sel(self):
        """ Tests associated to the .sel method """      
        # passing nothing or empty dict returns the complete NetworkSet
        self.assertEqual(self.ns_params.sel(), self.ns_params)
        self.assertEqual(self.ns_params.sel({}), self.ns_params)
        
        # should pass a dictionnary
        self.assertRaises(TypeError, self.ns_params.sel, 'wrong')
        self.assertRaises(TypeError, self.ns_params.sel, 1)
        
        # searching for a parameter which do not exist returns empty networkset
        self.assertEqual(self.ns.sel({'a': 1}), rf.NetworkSet())
        self.assertEqual(self.ns_params.sel({'ho ho': 1}), rf.NetworkSet())
        self.assertEqual(self.ns_params.sel({'a': 10}), rf.NetworkSet())        
        
        # there is two times the param key/value 'a':1 
        self.assertEqual(len(self.ns_params.sel({'a': 1})), 2)
        # Iterable values
        self.assertEqual(len(self.ns_params.sel({'a': [0,1]})), 4)
        self.assertEqual(len(self.ns_params.sel({'a': range(0,2)})), 4)
        # Multiple parameters
        self.assertEqual(len(self.ns_params.sel({'a': 0, 'X': 10})), 1)
        self.assertEqual(len(self.ns_params.sel({'a': 0, 'X': [10,20]})), 2)
        self.assertEqual(len(self.ns_params.sel({'a': [0,1], 'X': [10,20]})), 4)
        
    def test_interpolate_from_params(self):
        """ Tests associated to the .interpolate_from_params method """
        ## error handling
        # param does not exist
        self.assertRaises(ValueError, self.ns_params.interpolate_from_params, 'duh!', 0)
        # param values should be bounded by bounded by existing param values
        self.assertRaises(ValueError, self.ns_params.interpolate_from_params, 'a', -1, {'X': 10})
        self.assertRaises(ValueError, self.ns_params.interpolate_from_params, 'a', 100, {'X': 10})
        # cannot interpolate string-valued param
        self.assertRaises(ValueError, self.ns_params.interpolate_from_params, 'c', 'duh!', {'X': 10})       
        # ambiguity: could interpolate a for X=10 or X=20...
        self.assertRaises(ValueError, self.ns_params.interpolate_from_params, 'a', 0.5)
   
        ## working cases
        # returns a Network ?
        self.assertIsInstance(self.ns_params.interpolate_from_params('a', 0.5, {'X':10}), rf.Network)
        
        # test interpolated values
        f1 = rf.Frequency(1, 1, 1)
        ntwk0 = rf.Network(frequency=f1, s=[[0]], params={'s': 0})
        ntwk1 = rf.Network(frequency=f1, s=[[1]], params={'s': 1})
        ns2 = rf.NetworkSet([ntwk0, ntwk1])
        self.assertTrue(np.all(ns2.interpolate_from_params('s', 0.3).s == 0.3))

    def test_params_values(self):
        """Test the dictionnary containing all parameters names and values"""
        # returns None when no parameters are defined in a NetworkSet
        self.assertEqual(self.ns.params_values, None)
        # return a dict when parameters have been defined
        self.assertIsInstance(self.ns_params.params_values, dict)

        values = self.ns_params.params_values
        for idx, ntwk in enumerate(self.ns_params):
            for key in ntwk.params:
                self.assertEqual(values[key][idx], ntwk.params[key])

    def test_from_mdif(self):
        """ Create NetworkSets from MDIF files """
        mdif_files = glob.glob(self.test_dir+'../io/tests/MDIF_CITI_MDL/test_*.mdf')
        for mdif_file in mdif_files:
            print(mdif_file)
            self.assertIsInstance(rf.NetworkSet.from_mdif(mdif_file), rf.NetworkSet)

    def test_to_mdif(self):
        """ Test is NetworkSet are equal after writing and reading to MDIF """
        test_file = '_test.mdif'
        
        # without parameters 
        self.ns.write_mdif(test_file)   
        ns = rf.NetworkSet.from_mdif(test_file)
        self.assertEqual(ns, self.ns)
                
        # with parameters but without passing explicitly the values
        self.ns_params.write_mdif(test_file)
        ns_params = rf.NetworkSet.from_mdif(test_file)
        self.assertEqual(ns_params, self.ns_params)

        # with parameters and passing explicitly values but not types
        self.ns_params.write_mdif(test_file,
                                  values=self.ns_params.params_values)
        ns_params = rf.NetworkSet.from_mdif(test_file)
        self.assertEqual(ns_params, self.ns_params)        

        # with parameters and passing explicitly types but not values
        self.ns_params.write_mdif(test_file,
                                  data_types=self.ns_params.params_types)
        ns_params = rf.NetworkSet.from_mdif(test_file)
        self.assertEqual(ns_params, self.ns_params)  

        # with parameters and passing explicitly values and types
        self.ns_params.write_mdif(test_file,
                                  values=self.ns_params.params_values, 
                                  data_types=self.ns_params.params_types)
        ns_params = rf.NetworkSet.from_mdif(test_file)
        self.assertEqual(ns_params, self.ns_params)
        os.remove(test_file)
        
    def test_from_citi(self):
        """ Create NetworkSets from CITI files """
        citi_files = glob.glob(self.test_dir+'../io/tests/MDIF_CITI_MDL/test_*.cti')
        for citi_file in citi_files:
            print(citi_file)
            self.assertIsInstance(rf.NetworkSet.from_citi(citi_file), rf.NetworkSet)                

suite = unittest.TestLoader().loadTestsFromTestCase(NetworkSetTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
