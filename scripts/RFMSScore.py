import pandas as pd
import numpy as np
import logging
import os

# Logging configuration
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

class RFMSCalculator:
    def __init__(self, df):
        """
        Initializes the RFMSCalculator with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing transactional data.
        """
        self.df = df
        logger.info("RFMSCalculator initialized with DataFrame of shape %s", df.shape)

    def _calculate_recency(self):
        """
        Computes the recency metric based on the number of days since the last recorded transaction.

        Returns
        -------
        pd.DataFrame
            DataFrame with the Recency feature added.
        """
        logger.info("Calculating Recency")
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        max_date = self.df['TransactionStartTime'].max()
        self.df['Recency'] = (max_date - self.df['TransactionStartTime']).dt.days
        logger.info("Recency calculation completed")
        return self.df

    def _calculate_frequency(self):
        """
        Calculates the frequency metric, representing the count of transactions per customer.

        Returns
        -------
        pd.DataFrame
            DataFrame with Frequency calculated for each customer.
        """
        logger.info("Calculating Frequency")
        frequency_df = self.df.groupby('CustomerId').agg(Frequency=('TransactionId', 'count')).reset_index()
        logger.info("Frequency calculation completed")
        return frequency_df

    def _calculate_monetary_value(self):
        """
        Computes the monetary value metric, representing the total transaction amounts per customer.

        Returns
        -------
        pd.DataFrame
            DataFrame with Monetary values calculated for each customer.
        """
        logger.info("Calculating Monetary Value")
        monetary_df = self.df.groupby('CustomerId').agg(Monetary=('Amount', 'sum')).reset_index()
        logger.info("Monetary Value calculation completed")
        return monetary_df
    def _calculate_rfms_label(self):
        """
        Assigns an RFMS label based on the calculated Recency, Frequency, and Monetary value metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame with an RFMS_Label assigned based on certain criteria.
        """
        logger.info("Calculating RFMS Label")

        # Define thresholds for 'Good' or 'Bad' classification
        recency_threshold = self.df['Recency'].median()
        frequency_threshold = self.df['Frequency'].median()
        monetary_threshold = self.df['Monetary'].median()

        # Assign RFMS_Label: 0 for 'Good', 1 for 'Bad'
        conditions = [
            (self.df['Recency'] <= recency_threshold) & 
            (self.df['Frequency'] >= frequency_threshold) & 
            (self.df['Monetary'] >= monetary_threshold)
        ]

        # Apply conditions and assign 0 ('Good') if conditions are met, else 1 ('Bad')
        self.df['RFMS_Label'] = np.where(conditions[0], 0, 1)

        logger.info("RFMS Label assignment completed")
        return self.df

    def calculate_rfms(self):
        """
        Calculates the Recency, Frequency, and Monetary value metrics (RFMS).

        Returns
        -------
        pd.DataFrame
            DataFrame with RFMS features (Recency, Frequency, and Monetary value) merged.
        """
        logger.info("Starting RFMS calculation")

        # Calculate Recency
        self._calculate_recency()

        # Calculate Frequency and Monetary Value
        freq_df = self._calculate_frequency()
        mon_value_df = self._calculate_monetary_value()

        # Merge Frequency and Monetary value data
        rfms_df = pd.merge(freq_df, mon_value_df, on='CustomerId')

        # Merge RFMS features with the original DataFrame
        self.df = pd.merge(self.df, rfms_df, on='CustomerId', how='left')

        # Assign RFMS_Label based on calculated Recency, Frequency, and Monetary value
        self._calculate_rfms_label()

        logger.info("RFMS calculation and labeling completed")
        return self.df


class Labeling:
    def __init__(self, df):
        """
        Initializes the Labeling class with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RFMS scores and labels.
        """
        self.df = df
        logger.info("Labeling initialized with DataFrame of shape %s", df.shape)

    def assign_good_bad_labels(self):
        """
        Assigns 'Good' and 'Bad' labels to users based on the RFMS score.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'Good' and 'Bad' labels assigned.
        """
        logger.info("Assigning Good/Bad labels")
        # Assign labels based on RFMS_Label (0 is 'Good', 1 is 'Bad')
        self.df['User_Label'] = np.where(self.df['RFMS_Label'] == 0, 'Good', 'Bad')
        logger.info("Label assignment completed")
        return self.df