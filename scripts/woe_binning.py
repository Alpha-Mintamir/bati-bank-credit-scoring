import pandas as pd
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def calc_woe(df, feature, target):
    logger.info(f"Calculating WoE for feature: {feature}")
    
    try:
        total_good = df[target].sum()
        total_bad = df[target].count() - total_good

        df_woe = df.groupby(feature)[target].agg(['count', 'sum'])
        df_woe['bad'] = df_woe['count'] - df_woe['sum']
        df_woe['good_dist'] = df_woe['sum'] / total_good
        df_woe['bad_dist'] = df_woe['bad'] / total_bad
        df_woe['WoE'] = np.log(df_woe['good_dist'] / df_woe['bad_dist'])
        
        logger.info(f"WoE calculation completed for feature: {feature}")
        return df_woe[['WoE']]
    except Exception as e:
        logger.error(f"Error calculating WoE for feature: {feature}, Error: {e}")
        raise



def plot_woe(woe_df, feature_name):
    logger.info(f"Plotting WoE for feature: {feature_name}")
    
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=woe_df.index, y=woe_df['WoE'], palette='viridis')
        plt.title(f'Weight of Evidence (WoE) for {feature_name}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(f'{feature_name} Bins', fontsize=12)
        plt.ylabel('WoE', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"WoE plot completed for feature: {feature_name}")
    except Exception as e:
        logger.error(f"Error plotting WoE for feature: {feature_name}, Error: {e}")
        raise