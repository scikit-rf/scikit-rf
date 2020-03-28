
import skrf as rf
import unittest


class HomoDictTest(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.h = rf.util.HomoDict({'a':'asdf','b':'ZZZZ'})

    def test_get_item(self):
        self.assertEqual(self.h['a'],'asdf')


    def test_call(self):
        self.assertEqual(self.h.upper()['a'],'ASDF')


    def test_boolean_mask(self):
        self.assertEqual(self.h[self.h.startswith('a')==True], 'asdf')


class HomoListTest(unittest.TestCase):
    '''
    '''
    def setUp(self):
        self.h = rf.util.HomoList(['asdf','ZZZZ'])

    def test_get_item(self):
        self.assertEqual(self.h[0],'asdf')


    def test_call(self):
        self.assertEqual(self.h.upper()[0],'ASDF')


    def test_boolean_mask(self):
        self.assertEqual(self.h[self.h.startswith('a')==True], 'asdf')
