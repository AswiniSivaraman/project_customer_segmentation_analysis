import pandas as pd
import logging
from src.data_preprocessing import clean_data, encode_data, check_for_outlier, store_cleaned_data
from utils.encoding_values import get_pre_encoded_mappings, save_encoded_mappings
from zenml import step

@step
def preprocessing_data(df: pd.DataFrame, categorical_columns: list, save_path: str) -> pd.DataFrame:
    """
    ZenML Step to preprocess the dataset:
    - Cleans the data (removes nulls, duplicates, and drops unnecessary columns).
    - Stores the cleaned dataset.
    - Encodes categorical features.
    - Checks for outliers.
    - Apply feature engineering transformations.
         - creates a new target feature for classification ('purchase_completed')
         - creates time-based, season-based, and session-based features.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        categorical_columns (list): List of categorical columns to encode.
        save_path (str): Path to save the cleaned dataset.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset.
    """
    
    try:
        logging.info(f'Starting data preprocessing.....')

        df_cleaned = clean_data(df)
        logging.info(f'Data cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.')

        encoding_mappings = {}
        for col in categorical_columns:
            df_cleaned, mapping = encode_data(df_cleaned,col)
            encoding_mappings[col] = mapping
        logging.info(f"Categorical columns encoded: {categorical_columns}")

        pre_encoded_mappings = get_pre_encoded_mappings()
        encoding_mappings.update(pre_encoded_mappings)
        logging.info(f"Merged encoded mappings for both manually encoded & pre-encoded columns.")

        mappings_file_path = save_encoded_mappings(encoding_mappings)
        logging.info(f"Encoded mappings saved at {mappings_file_path}")

        outlier_report = check_for_outlier(df_cleaned, column=None)
        logging.info(f"Outlier report generated:\n{outlier_report}")

        return df_cleaned
        
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        logging.exception("Full Exception Traceback:")
        raise e

