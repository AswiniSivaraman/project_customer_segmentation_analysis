import pandas as pd
import logging
from zenml import step
from utils.data_transformation import apply_standard_scaling, handle_imbalanced_data
from typing import Tuple

@step
def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, numerical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML step to apply Standard Scaling on both train & test datasets using train data to fit the scaler.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        numerical_cols (list): List of numerical columns to scale.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test datasets.
    """
    try:
        logging.info("Starting Standard Scaling process...")
        train_scaled, test_scaled = apply_standard_scaling(train_df, test_df, numerical_cols)
        logging.info("Standard Scaling successfully applied on train & test data.")
        return train_scaled, test_scaled

    except Exception as e:
        logging.error(f"Error in feature scaling using StandardScalar step: {e}")
        logging.exception('Full Exception Traceback:')
        raise e


@step
def balance_train_data(train_df: pd.DataFrame, target_col: str, method: str = "smote") -> Tuple[pd.DataFrame, pd.Series]:
    """
    ZenML step to handle imbalanced data **only on training data** using SMOTE or undersampling.

    Args:
        train_df (pd.DataFrame): The training dataset.
        target_col (str): The target column for classification.
        method (str): Either `"smote"` for oversampling or `"undersampling"` for reducing majority class.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced training dataset (X_train, y_train).
    """
    try:
        logging.info(f"Starting Data Balancing process using {method}...")
        X_train_balanced, y_train_balanced = handle_imbalanced_data(train_df, target_col)
        logging.info("Data balancing completed successfully.")
        return X_train_balanced, y_train_balanced

    except Exception as e:
        logging.error(f"Error in data balancing using SMOTE step: {e}")
        logging.exception('Full Exception Traceback:')
        raise e
