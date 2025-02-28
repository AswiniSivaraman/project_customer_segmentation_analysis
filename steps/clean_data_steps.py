import pandas as pd
import logging
from src.data_preprocessing import clean_data, encode_data, check_for_outlier, store_cleaned_data
from utils.encoding_values import get_pre_encoded_mappings, save_encoded_mappings
from zenml import step
from typing import Tuple

@step
def clean_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML Step to clean the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    try:
        logging.info(f'Starting data cleaning.....')
        df_cleaned = clean_data(df)
        logging.info(f'Data cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.')
        return df_cleaned
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        logging.exception("Full Exception Traceback:")
        raise e
    

@step
def encode_data_step(df_train: pd.DataFrame, df_test: pd.DataFrame, categorical_columns: list, save_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML Step to encode categorical features for both train and test data.
    
    Args:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        categorical_columns (list): List of categorical columns to encode.
        save_path (str): Path to save encoded mappings.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and test datasets.
    """
    try:
        logging.info("Starting data encoding for train and test datasets...")
        encoding_mappings = get_pre_encoded_mappings()

        # Apply encoding to both train and test data
        for col in categorical_columns:
            df_train, mapping = encode_data(df_train, col, is_train=True, save_path=save_path)  # Fit & Save Encoder
            df_test, _ = encode_data(df_test, col, is_train=False, save_path=save_path)  # Load & Transform Test Data
            encoding_mappings[col] = mapping

        save_encoded_mappings(encoding_mappings, save_path)
        logging.info(f"Categorical columns encoded: {categorical_columns}")

        return df_train, df_test

    except Exception as e:
        logging.error(f"Error in encoding data: {e}")
        logging.exception("Full Exception Traceback:")
        raise e


@step
def check_outlier_step(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    ZenML Step to check for outliers.
    
    Args:
        df (pd.DataFrame): Dataset with encoded categorical features.
        columns (list, optional): List of specific columns to check for outliers. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing outlier information.
    """
    try:
        logging.info(f'Starting outlier detection.....')
        outlier_report = check_for_outlier(df, columns)
        return outlier_report
    except Exception as e:
        logging.error(f"Error in detecting outliers: {e}")
        logging.exception("Full Exception Traceback:")
        raise e