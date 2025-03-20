# steps/rfe_feature_selection.py
import pandas as pd
import logging
from zenml import step
from utils.rfe_selection import rfe_feature_selection
from typing import Tuple


@step
def apply_rfe(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, n_features: int = 9, estimator=None) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    ZenML step to apply RFE on the training data and filter both training and test data 
    to only include the selected features (plus the target column).
    
    Args:
        train_df (pd.DataFrame): Scaled training data (including target).
        test_df (pd.DataFrame): Scaled test data (including target).
        target_column (str): Name of the target column.
        n_features (int): Number of features to select.
        estimator: A scikit-learn estimator to use for RFE. Defaults to LinearRegression() if None.
    
    Returns:
        Tuple:
          - train_filtered (pd.DataFrame): Training data filtered to selected features plus target.
          - test_filtered (pd.DataFrame): Test data filtered to selected features plus target.
          - selected_features (list): List of selected feature names.
    """
    try:
        selected_features = rfe_feature_selection(train_df, target_column, n_features, estimator)
        logging.info(f"RFE selected features: {selected_features}")
        
        # Filter both train and test data to include only the selected features and the target column.
        train_filtered = train_df[selected_features + [target_column]]
        logging.info(f'filtered columns --> {train_filtered.columns}')
        test_filtered = test_df[selected_features + [target_column]]
        return train_filtered, test_filtered, selected_features
    except Exception as e:
        logging.error("Error applying RFE")
        raise e
