import pandas as pd
import logging
from zenml import step
from steps.clean_data_steps import clean_data_step, encode_data_step, check_outlier_step
from steps.feature_engineering import apply_feature_engineering

@step
def preprocessing_data(df: pd.DataFrame, cat_columns: list, columns: list, save_path: str) -> pd.DataFrame:
    """
    ZenML Step to preprocess the dataset:
    - Cleans the data.
    - Encodes categorical features.
    - Checks for outliers.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        categorical_columns (list): List of categorical columns to encode.
        save_path (str): Path to save encoded mappings.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset.
    """
    try:
        logging.info(f'Starting full data preprocessing.....')

        # Clean the data
        df_cleaned = clean_data_step(df)
        logging.info(f'Data cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns.')

        # Apply feature engineering
        df_featured = apply_feature_engineering(df_cleaned)
        logging.info(f"Feature engineering applied.")
        logging.info(f"Featured data columns: {df_featured.columns}")

        # Encode categorical features
        df_encoded = encode_data_step(df_featured, cat_columns, save_path)
        logging.info(f"Categorical columns encoded: {columns}")
        logging.info(f"Encoded data columns: {df_encoded.columns}")

        # Check for outliers
        outlier_report = check_outlier_step(df_encoded, columns)
        logging.info(f"Outlier report generated:\n{outlier_report}")

        return df_encoded
        
    except Exception as e:
        logging.error(f"Error in full data preprocessing: {e}")
        logging.exception("Full Exception Traceback:")
        raise e
