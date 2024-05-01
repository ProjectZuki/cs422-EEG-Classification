"""
NOTE: Still in progress
"""

import unittest
import pandas as pd
import train as train

class TestDataImputation(unittest.TestCase):

    def setUp(self):
        # create sample DataFrame with missing values
        self.data_with_missing = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 5, 6, None],
            'C': [7, None, None, 10]
        })

    def test_impute_null(self):
        # sample imputation
        imputed_data = train.impute_null(self.data_with_missing)
        
        # check if there are still missing values after imputation
        self.assertFalse(imputed_data.isnull().any().any())
        
        # check shape of imputed data to match original
        self.assertEqual(imputed_data.shape, self.data_with_missing.shape)
        
if __name__ == '__main__':
    unittest.main()
