import unittest
import pandas as pd
from scripts.feature_engineering import AggregateFeatures, Extracting_features, normalize_numerical_features

class TestAggregateFeatures(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 150, 200, 300, 400],
            'TransactionId': [1, 2, 3, 4, 5]
        }
        self.df = pd.DataFrame(data)
        self.aggregate_features = AggregateFeatures(self.df)

    def test_sum_all_transactions(self):
        self.aggregate_features.sum_all_transactions()
        expected = [250, 500, 400]  # Total amounts for CustomerId 1, 2, and 3
        result = self.aggregate_features.get_dataframe()['TotalTransactionAmount'].tolist()
        self.assertEqual(result, expected)

    def test_average_transaction_amount(self):
        self.aggregate_features.average_transaction_amount()
        expected = [125.0, 250.0, 400.0]  # Average amounts for CustomerId 1, 2, and 3
        result = self.aggregate_features.get_dataframe()['AverageTransactionAmount'].tolist()
        self.assertEqual(result, expected)

    def test_transaction_count(self):
        self.aggregate_features.transaction_count()
        expected = [2, 2, 1]  # Counts for CustomerId 1, 2, and 3
        result = self.aggregate_features.get_dataframe()['TotalTransactions'].tolist()
        self.assertEqual(result, expected)

    def test_standard_deviation_amount(self):
        self.aggregate_features.standard_deviation_amount()
        expected = [25.0, 70.71, None]  # Std deviation for CustomerId 1, 2, and NaN for 3
        result = self.aggregate_features.get_dataframe()['StdTransactionAmount'].tolist()
        self.assertAlmostEqual(result[0], expected[0], places=2)
        self.assertAlmostEqual(result[1], expected[1], places=2)
        self.assertIsNone(result[2])  # CustomerId 3 should have no transactions

class TestExtractingFeatures(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'TransactionId': [1, 2, 3],
            'TransactionStartTime': ['2023-10-01 12:00:00', '2023-10-02 13:30:00', '2023-10-03 15:45:00']
        }
        self.df = pd.DataFrame(data)
        self.extracting_features = Extracting_features(self.df)

    def test_transaction_hour(self):
        self.extracting_features.transaction_hour()
        expected = [12, 13, 15]  # Expected hours extracted
        result = self.extracting_features.get_dataframe()['TransactionHour'].tolist()
        self.assertEqual(result, expected)

    def test_transaction_day(self):
        self.extracting_features.transaction_day()
        expected = [1, 2, 3]  # Expected days extracted
        result = self.extracting_features.get_dataframe()['TransactionDay'].tolist()
        self.assertEqual(result, expected)

    def test_transaction_month(self):
        self.extracting_features.transaction_month()
        expected = [10, 10, 10]  # Expected months extracted
        result = self.extracting_features.get_dataframe()['TransactionMonth'].tolist()
        self.assertEqual(result, expected)

    def test_transaction_year(self):
        self.extracting_features.transaction_year()
        expected = [2023, 2023, 2023]  # Expected years extracted
        result = self.extracting_features.get_dataframe()['TransactionYear'].tolist()
        self.assertEqual(result, expected)

class TestNormalizeNumericalFeatures(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'TransactionId': [1, 2, 3],
            'Amount': [100, 150, 200]
        }
        self.df = pd.DataFrame(data)

    def test_standardize(self):
        result_df = normalize_numerical_features(self.df.copy(), ['Amount'], method='standardize')
        self.assertAlmostEqual(result_df['Amount'].mean(), 0, places=2)
        self.assertAlmostEqual(result_df['Amount'].std(), 1, places=2)

    def test_normalize(self):
        result_df = normalize_numerical_features(self.df.copy(), ['Amount'], method='normalize')
        self.assertAlmostEqual(result_df['Amount'].min(), 0, places=2)
        self.assertAlmostEqual(result_df['Amount'].max(), 1, places=2)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            normalize_numerical_features(self.df.copy(), ['Amount'], method='invalid')

if __name__ == '__main__':
    unittest.main()
