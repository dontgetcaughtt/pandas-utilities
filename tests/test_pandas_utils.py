import unittest

import numpy as np
import pandas as pd

from pdutils import eq_multiple, equals_multiple

class TestMultipleEqual (unittest.TestCase):
    
    def setUp (self):
        self.s1 = pd.Series([1.0, 2.0,    3.0])
        self.s2 = pd.Series([1.0, np.nan, 3.0])
        self.s3 = pd.Series([1.0, np.nan, 3.0])
        self.s4 = pd.Series([1.0, -2.0,   3.0])
        self.s5 = pd.Series([1.0, np.nan, np.nan])
        
        self.l2 = list(self.s2)
        self.t2 = tuple(self.s2)
        self.n2 = self.s2.values
        self.d2 = dict(self.s2)
    
    def test_multiple_eq (self):
        r = eq_multiple([self.s1, self.s2])
        self.assertTrue(isinstance(r, pd.Series))
        self.assertEqual(len(r), 3)
        
        # Series w/ and w/o NaN
        self.assertEqual(sum(eq_multiple([self.s1, self.s2], na='all')),  2)
        self.assertEqual(sum(eq_multiple([self.s1, self.s2], na='any')),  3)
        self.assertEqual(sum(eq_multiple([self.s1, self.s2], na='none')), 2)
        
        # all Series w/ NaN
        self.assertEqual(sum(eq_multiple([self.s2, self.s2], na='all')),  3)
        self.assertEqual(sum(eq_multiple([self.s2, self.s2], na='any')),  3)
        self.assertEqual(sum(eq_multiple([self.s2, self.s2], na='none')), 2)
        
        # all Series w/o NaN
        self.assertEqual(sum(eq_multiple([self.s1, self.s1], na='all')),  3)
        self.assertEqual(sum(eq_multiple([self.s1, self.s1], na='any')),  3)
        self.assertEqual(sum(eq_multiple([self.s1, self.s1], na='none')), 3)
        
        # all/none check equality pairwise, and may not be obscured by interjacent NaNs
        self.assertEqual(sum(eq_multiple([self.s1, self.s2, self.s3, self.s4], na='all')), 2)
        self.assertEqual(sum(eq_multiple([self.s1, self.s2, self.s3, self.s4], na='any')), 2)
        self.assertEqual(sum(eq_multiple([self.s1, self.s2, self.s3, self.s4], na='none')), 2)
        
        self.assertEqual(sum(eq_multiple([self.s5, self.s2, self.s3], na='all')),  2)
        self.assertEqual(sum(eq_multiple([self.s5, self.s2, self.s3], na='any')),  3)
        self.assertEqual(sum(eq_multiple([self.s5, self.s2, self.s3], na='none')), 1)
        
        # order of NaN could theoretically affect pairwise comparison
        a = eq_multiple([self.s1, self.s2, self.s3, self.s4], na='any')
        b = eq_multiple([self.s1, self.s4, self.s3, self.s2], na='any')
        self.assertTrue(pd.Series.eq(a, b).all())
        a = eq_multiple([self.s1, self.s2, self.s3], na='any')
        b = eq_multiple([self.s2, self.s1, self.s3], na='any')
        self.assertTrue(pd.Series.eq(a, b).all())
        
        # single elements are always equal
        self.assertEqual(sum(eq_multiple([self.s5], na='all')),  3)
        self.assertEqual(sum(eq_multiple([self.s5], na='any')),  3)
        self.assertEqual(sum(eq_multiple([self.s5], na='none')), 3)
        
        self.assertEqual(sum(eq_multiple(self.s5, na='all')),  3)
        self.assertEqual(sum(eq_multiple(self.s5, na='any')),  3)
        self.assertEqual(sum(eq_multiple(self.s5, na='none')), 3)
        self.assertTrue(isinstance(eq_multiple(self.s5), pd.Series))
        
        # empty list
        self.assertRaises(ValueError, eq_multiple, [])
        
        # Non-Series
        self.assertEqual(sum(eq_multiple([self.s5, self.l2, self.s3], na='all')), 2)
        self.assertEqual(sum(eq_multiple([self.s5, self.t2, self.s3], na='all')), 2)
        self.assertEqual(sum(eq_multiple([self.s5, self.n2, self.s3], na='all')), 2)
        self.assertRaises(TypeError, eq_multiple, [self.s5, self.d2, self.s3], na='all')
        
        # invalid keyword argument
        self.assertRaises(ValueError, eq_multiple, [self.s1, self.s2], 'foobar')
        
        # DataFrame as input
        df = pd.concat([self.s5, self.s2, self.s3], axis='columns')
        self.assertEqual(sum(eq_multiple(df, na='all')), 2)
        self.assertEqual(sum(eq_multiple(df, na='any')), 3)
        self.assertEqual(sum(eq_multiple(df, na='none')), 1)
    
    def test_multiple_equals (self):
        r = equals_multiple([self.s1, self.s2])
        self.assertTrue(isinstance(r, np.bool_))
        
        # Series w/ and w/o NaN
        self.assertFalse(equals_multiple([self.s1, self.s2], na='all'))
        self.assertTrue(equals_multiple([self.s1, self.s2], na='any'))
        self.assertFalse(equals_multiple([self.s1, self.s2], na='none'))
        
        # all Series w/ NaN
        self.assertTrue(equals_multiple([self.s2, self.s2], na='all'))
        self.assertTrue(equals_multiple([self.s2, self.s2], na='any'))
        self.assertFalse(equals_multiple([self.s2, self.s2], na='none'))
        
        # all Series w/o NaN
        self.assertTrue(equals_multiple([self.s1, self.s1], na='all'))
        self.assertTrue(equals_multiple([self.s1, self.s1], na='any'))
        self.assertTrue(equals_multiple([self.s1, self.s1], na='none'))
        
        # all/none check equality pairwise, and may not be obscured by interjacent NaNs
        self.assertFalse(equals_multiple([self.s1, self.s2, self.s3, self.s4], na='all'))
        self.assertFalse(equals_multiple([self.s1, self.s2, self.s3, self.s4], na='any'))
        self.assertFalse(equals_multiple([self.s1, self.s2, self.s3, self.s4], na='none'))
        
        self.assertFalse(equals_multiple([self.s5, self.s2, self.s3], na='all'))
        self.assertTrue(equals_multiple([self.s5, self.s2, self.s3], na='any'))
        self.assertFalse(equals_multiple([self.s5, self.s2, self.s3], na='none'))
        
        # order of NaN could theoretically affect pairwise comparison
        a = equals_multiple([self.s1, self.s2, self.s3, self.s4], na='any')
        b = equals_multiple([self.s1, self.s4, self.s3, self.s2], na='any')
        self.assertTrue(a == b)
        a = equals_multiple([self.s1, self.s2, self.s3], na='any')
        b = equals_multiple([self.s2, self.s1, self.s3], na='any')
        self.assertTrue(a == b)
        
        # single elements are always equal
        self.assertTrue(equals_multiple([self.s5], na='all'))
        self.assertTrue(equals_multiple([self.s5], na='any'))
        self.assertTrue(equals_multiple([self.s5], na='none'))
        
        self.assertTrue(equals_multiple(self.s5, na='all'))
        self.assertTrue(equals_multiple(self.s5, na='any'))
        self.assertTrue(equals_multiple(self.s5, na='none'))
        
        # empty list
        self.assertRaises(ValueError, equals_multiple, [])
        
        # Non-Series
        self.assertFalse(equals_multiple([self.s5, self.l2, self.s3], na='all'))
        self.assertFalse(equals_multiple([self.s5, self.t2, self.s3], na='all'))
        self.assertFalse(equals_multiple([self.s5, self.n2, self.s3], na='all'))
        self.assertRaises(TypeError, equals_multiple, [self.s5, self.d2, self.s3], na='all')
        
        # invalid keyword argument
        self.assertRaises(ValueError, equals_multiple, [self.s1, self.s2], 'foobar')

        # DataFrame as input
        df = pd.concat([self.s5, self.s2, self.s3], axis='columns')
        self.assertFalse(equals_multiple(df, na='all'))
        self.assertTrue(equals_multiple(df, na='any'))
        self.assertFalse(equals_multiple(df, na='none'))

if __name__ == '__main__':
    unittest.main()