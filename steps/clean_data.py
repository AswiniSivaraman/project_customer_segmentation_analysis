import pandas as pd
import logging
from zenml import step
from steps.clean_data_steps import clean_data_step, encode_data_step, check_outlier_step
from steps.feature_engineering import apply_feature_engineering
from typing import Tuple

@step
def preprocessing_data(df_train: pd.DataFrame, df_test: pd.DataFrame, cat_columns: list, columns: list, save_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML Step to preprocess the dataset:
    - Cleans the data.
    - Encodes categorical features.
    - Checks for outliers.
    
    Args:
        df_train (pd.DataFrame): Raw training dataset.
        df_test (pd.DataFrame): Raw testing dataset.
        cat_columns (list): List of categorical columns to encode.
        save_path (str): Path to save encoded mappings.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned and preprocessed training and testing datasets.
    """
    try:
        logging.info("Starting full data preprocessing...")

        # Clean the data
        df_train_cleaned = clean_data_step(df_train)
        df_test_cleaned = clean_data_step(df_test)
        logging.info(f"Data cleaned: Train -> {df_train_cleaned.shape}, Test -> {df_test_cleaned.shape}")

        # Apply feature engineering
        df_train_featured = apply_feature_engineering(df_train_cleaned)
        df_test_featured = apply_feature_engineering(df_test_cleaned)
        logging.info("Feature engineering applied.")

        # Encode categorical features (Updated to handle train & test separately)
        df_train_encoded, df_test_encoded = encode_data_step(df_train_featured, df_test_featured, cat_columns, save_path)
        logging.info(f"Categorical columns encoded: {columns}")

        return df_train_encoded, df_test_encoded
        
    except Exception as e:
        logging.error(f"Error in full data preprocessing: {e}")
        logging.exception("Full Exception Traceback:")
        raise e
