import pandas as pd
import logging
from src.data_preprocessing import clean_data, encode_data, check_for_outlier, store_cleaned_data
from utils.encoding_values import get_pre_encoded_mappings, save_encoded_mappings
from zenml import step

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
def encode_data_step(df: pd.DataFrame, categorical_columns: list, save_path: str) -> pd.DataFrame:
    """
    ZenML Step to encode categorical features.
    
    Args:
        df (pd.DataFrame): Cleaned dataset.
        categorical_columns (list): List of categorical columns to encode.
        save_path (str): Path to save encoded mappings.
    
    Returns:
        pd.DataFrame: Dataset with encoded categorical features.
    """
    try:
        logging.info(f'Starting data encoding.....')
        encoding_mappings = {}
        for col in categorical_columns:
            df, mapping = encode_data(df, col)
            encoding_mappings[col] = mapping
        logging.info(f"Categorical columns encoded: {categorical_columns}")

        pre_encoded_mappings = get_pre_encoded_mappings()
        encoding_mappings.update(pre_encoded_mappings)
        logging.info(f"Merged encoded mappings for both manually encoded & pre-encoded columns.")

        mappings_file_path = save_encoded_mappings(encoding_mappings, save_path)
        logging.info(f"Encoded mappings saved at {mappings_file_path}")
        
        return df
    except Exception as e:
        logging.error(f"Error in encoding data: {e}")
        logging.exception("Full Exception Traceback:")
        raise


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