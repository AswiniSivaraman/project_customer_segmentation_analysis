import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import os
import pickle

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean the data by removing the null and duplicate values and dropping the unnecessary columns
    and returned the cleaned dataframe

    Args:
    df --> pandas dataframe

    Returns:
    pandas dataframe

    """
    try:
        # Check for null values
        print("Null values in the dataset:")
        print(df.isnull().sum())

        # Check for duplicates
        print("Duplicate rows in the dataset:")
        print(df.duplicated().sum())

        # Drop the unnecessary columns
        cols_to_drop = ['year']
        df = df.drop(cols_to_drop, axis=1)
        print('Columns dropped:', cols_to_drop)

        logging.info(f'Cleaning the data by removing the null, duplicate values and dropping the unnecessary columns')
        return df.dropna().drop_duplicates()
    
    except Exception as e:
        logging.error(f'Error occured when cleaning the data --> : {e}')
        raise e
    


def encode_data(df: pd.DataFrame, col_name: str, is_train: bool, save_path: str, pipeline: str) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical data using LabelEncoder. Only return encoded data and mapping,
    as mappings are handled separately in save_encoded_mappings.

    Args:
    df (pd.DataFrame): DataFrame to encode.
    col_name (str): Column name to encode.
    is_train (bool): Flag indicating if data is training data.
    save_path (str): Directory to save the encoder file.
    pipeline (str): Name of the pipeline (e.g., 'regression', 'classification', 'clustering').

    Returns:
    Tuple[pd.DataFrame, dict]: DataFrame with encoded column and mapping dictionary.
    """
    try:
        logging.info(f"Encoding column: {col_name} in pipeline: {pipeline}")
        
        add_ons_path = os.path.join(save_path, "add_ons")
        os.makedirs(add_ons_path, exist_ok=True)  # Ensure add_ons directory exists
        mapping_path = os.path.join(add_ons_path, f"{pipeline}_{col_name}_mapping.pkl")
        
        if is_train:
            le = LabelEncoder()
            df[col_name] = le.fit_transform(df[col_name])
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            
            # Save the mapping
            with open(mapping_path, "wb") as file:
                pickle.dump(mapping, file)
            logging.info(f"Label Encoding Mapping for {col_name} saved at {mapping_path}")
        else:
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Mapping file for {col_name} not found: {mapping_path}")
            
            logging.info(f"Loading Label Encoding Mapping for {col_name} from {mapping_path}")
            with open(mapping_path, "rb") as file:
                mapping = pickle.load(file)
            
            df[col_name] = df[col_name].apply(lambda x: mapping[x] if x in mapping else -1)
        
        return df, mapping
    
    except Exception as e:
        logging.error(f"Error occurred when encoding {col_name} in {pipeline}: {e}")
        logging.exception("Full Exception Traceback:")
        raise e



def store_cleaned_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Store the cleaned data to a csv file

    Args:
    df --> pandas dataframe
    file_path --> path to store the cleaned data

    Returns:
    None
    """
    try:
        logging.info(f'Storing the cleaned data to a csv file --> {file_path}') 
        df.to_csv(file_path, index=False)
    except Exception as e:
        logging.error(f'Error occured when storing the cleaned data --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e
    


def check_for_outlier(df, columns=None):
    """
    Check for outliers in the numerical columns of the dataframe and return a dataframe containing outliers count, lower bound, upper bound, min and max values.

    Args:
    df --> pandas dataframe
    columns (optional) --> list of column names to check for outliers. If None, checks all numerical columns.

    Returns:
    pandas dataframe
    """
    try:
        logging.info(f'Checking for outliers in the numerical columns of the dataframe')
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_df = []
        if columns:  # If a list of specific columns is provided
            numerical_cols = columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_df.append({'col': col, 'outliers_count': len(outliers), 'lower bound': lower_bound, 'upper bound': upper_bound, 'min value': df[col].min(), 'max_value': df[col].max()})
        return pd.DataFrame(outlier_df)
    except Exception as e:
        logging.error(f'Error occurred when checking for outliers --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e
