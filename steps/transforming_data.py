import pandas as pd
import logging
from zenml import step
from utils.data_transformation import apply_standard_scaling, handle_imbalanced_data
from typing import Tuple


@step
def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, numerical_cols: list, target_column: str, is_target_there: bool, pipeline_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    ZenML step to apply Standard Scaling on both train & test datasets using train data to fit the scaler.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        numerical_cols (list): List of numerical columns to scale.
        target_column (str): The target column (excluded from scaling).
        pipeline (str): Name of the pipeline (e.g., 'regression', 'classification', 'clustering').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test datasets (target remains unchanged).
    """
    try:
        logging.info("Starting Standard Scaling process...")
        train_scaled, test_scaled = apply_standard_scaling(train_df, test_df, numerical_cols, target_column, is_target_there, pipeline_type)
        feature_list = list(train_scaled.columns)
        print(f"feature_list data type -- >{type(feature_list)}")
        print(f"{pipeline_type} scaled columns --> {feature_list}")
        logging.info("Standard Scaling successfully applied on train & test data.")
        return train_scaled, test_scaled, feature_list

    except Exception as e:
        logging.error(f"Error in feature scaling using StandardScaler step: {e}")
        logging.exception("Full Exception Traceback:")
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
        print("x_train_balance type", type(X_train_balanced))
        print("y_test_balance type", type(y_train_balanced))
        return X_train_balanced, y_train_balanced

    except Exception as e:
        logging.error(f"Error in data balancing using SMOTE step: {e}")
        logging.exception('Full Exception Traceback:')
        raise e
