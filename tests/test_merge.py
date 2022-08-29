import unittest

import numpy as np
import pandas as pd

from pdutils import merge_non_na

class TestMergeNonNA(unittest.TestCase):
    
    def setUp (self):
        self.df1 = pd.DataFrame({'A': [1.0, np.nan],    'B': [np.nan, np.nan], 'C': [1.0, 2.0]})
        self.df2 = pd.DataFrame({'A': [1.0, 2.0],       'B': [np.nan, np.nan], 'C': [np.nan, 2.0]})
        self.df3 = pd.DataFrame({'B': [np.nan, np.nan], 'A': [np.nan, 2.0],    'C': [1.0, 9.9]})
        self.df4 = pd.DataFrame({'A': [1.0, np.nan],    'B': [np.nan, np.nan], 'C': [1.0, np.nan]})
        self.df  = pd.DataFrame({'A': [1.0, 2.0]})
        
        self.s1 = pd.Series({'red': np.nan, 'blue': 222.2,  'green': np.nan})
        self.s2 = pd.Series({'red': 777.7,  'blue': np.nan, 'green': np.nan})
        self.s3 = pd.Series({'red': 777.7,  'blue': 222.2,  'green': np.nan})
        self.s4 = pd.Series({'red': 777.7,  'blue': np.nan, 'green': 111.1})
        self.s5 = pd.Series({'red': 777.7,  'blue': 222.2,  'green': 456.7})
        self.s6 = pd.Series({'red': 777.7,                  'green': 111.1})
        self.s7 = pd.Series({'red': 777.7,  'cyan': 135.7,  'green': 111.1})
        self.s8 = pd.Series({'red': np.nan, 'blue': np.nan, 'green': np.nan})
        self.s  = pd.Series({'red': 777.7,  'blue': 222.2,  'green': 111.1})
        
        self.l1 = list(self.s1)
        self.t1 = tuple(self.s1)
        self.n1 = self.s1.values
        self.d1 = dict(self.s1)
        self.sx = self.s4.reset_index(drop=True)
        self.s0 = self.s.reset_index(drop=True)
    
    def test_series (self):
        r = merge_non_na([self.s1, self.s4])
        self.assertTrue(isinstance(r, pd.Series))
        self.assertEqual(len(self.s1), len(r))
        
        r = merge_non_na([self.s1, self.s2, self.s3, self.s4])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # order does not matter
        r = merge_non_na([self.s2, self.s1, self.s4, self.s3])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # do not override with new NaNs
        r = merge_non_na([self.s1, self.s4])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # ignore unequal values
        r = merge_non_na([self.s1, self.s4, self.s5])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # missing index
        r = merge_non_na([self.s1, self.s6, self.s3])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # all NaN slice
        r = merge_non_na([self.s8, self.s])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # missing index
        r = merge_non_na([self.s1, self.s6])
        self.assertTrue(pd.Series.equals(r, self.s))
        
        # merge and expand
        r = merge_non_na([self.s1, self.s7])
        m = pd.Series({'red': 777.7, 'blue': 222.2, 'green': 111.1, 'cyan': 135.7})
        self.assertTrue(pd.Series.equals(r, m))
        
        # keep missing NaN
        r = merge_non_na([self.s2, self.s6], remaining_na='ignore')
        m = pd.Series({'red': 777.7, 'blue': np.nan, 'green': 111.1})
        self.assertTrue(pd.Series.equals(r, m))
        
        # keep missing NaN and expand
        r = merge_non_na([self.s2, self.s7], remaining_na='ignore')
        m = pd.Series({'red': 777.7, 'blue': np.nan, 'green': 111.1, 'cyan': 135.7})
        self.assertTrue(pd.Series.equals(r, m))
        
        # single element
        r = merge_non_na([self.s])
        self.assertTrue(isinstance(r, pd.Series))
        self.assertEqual(len(self.s), len(r))
        
        # empty list
        self.assertRaises(ValueError, merge_non_na, [])
    
    def test_frames (self):
        self.assertTrue(isinstance(merge_non_na(self.df), pd.Series))
        
        r = merge_non_na(self.df1)
        self.assertTrue(pd.Series.equals(r, self.df['A']))
        
        # do not overwrite with new NaNs
        r = merge_non_na(self.df2)
        self.assertTrue(pd.Series.equals(r, self.df['A']))
        
        # ignore unequal values
        r = merge_non_na(self.df3)
        self.assertTrue(pd.Series.equals(r, self.df['A']))
    
    def test_mixed (self):
        # list, tuple and numpy array have no index
        r = merge_non_na([self.l1, self.sx])
        self.assertTrue(pd.Series.equals(r, self.s0))
        
        r = merge_non_na([self.t1, self.sx])
        self.assertTrue(pd.Series.equals(r, self.s0))
        
        r = merge_non_na([self.n1, self.sx])
        self.assertTrue(pd.Series.equals(r, self.s0))
        
        # dict does have an index
        r = merge_non_na([self.d1, self.s4])
        self.assertTrue(pd.Series.equals(r, self.s))
    
    def test_remaining (self):
        with self.assertRaises(ValueError):
            merge_non_na(self.df4)
        
        with self.assertRaises(ValueError):
            merge_non_na([self.s1, self.s2, self.s3], remaining_na='raise')
        
        with self.assertWarns(UserWarning):
            merge_non_na([self.s1, self.s2, self.s3], remaining_na='warn')
        
        r = merge_non_na([self.s1, self.s2, self.s3], remaining_na='ignore')
        self.assertTrue(pd.Series.equals(r, self.s3))
        
        with self.assertRaises(ValueError):
            merge_non_na([self.s1, self.s2, self.s3], remaining_na='foobar')

if __name__ == '__main__':
    unittest.main()