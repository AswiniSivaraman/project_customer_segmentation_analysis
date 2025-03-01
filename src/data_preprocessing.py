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
        cols_to_drop = ["year"]
        df = df.drop(cols_to_drop, axis=1)
        print('Columns dropped:', cols_to_drop)

        logging.info(f'Cleaning the data by removing the null, duplicate values and dropping the unnecessary columns')
        return df.dropna().drop_duplicates()
    
    except Exception as e:
        logging.error(f'Error occured when cleaning the data --> : {e}')
        raise e
    


def encode_data(df: pd.DataFrame, col_name: str, is_train: bool, save_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical data using LabelEncoder. Save the encoder when processing training data,
    and load & apply it for test data.

    Args:
    df (pd.DataFrame): DataFrame to encode.
    col_name (str): Column name to encode.
    is_train (bool): Flag indicating if data is training data.
    save_path (str): Directory to save the encoder file.

    Returns:
    Tuple[pd.DataFrame, dict]: DataFrame with encoded column and mapping dictionary.
    """
    try:
        encoder_path = os.path.join(save_path, f"{col_name}_encoder.pkl")
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

        if is_train:
            logging.info(f"Fitting LabelEncoder for {col_name} and saving to {encoder_path}")
            le = LabelEncoder()
            df[col_name] = le.fit_transform(df[col_name])

            # Save the encoder object
            with open(encoder_path, "wb") as file:
                pickle.dump(le, file)

        else:
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder file for {col_name} not found: {encoder_path}")

            logging.info(f"Loading LabelEncoder for {col_name} from {encoder_path}")
            with open(encoder_path, "rb") as file:
                le = pickle.load(file)

            # df[col_name] = le.transform(df[col_name])
            df[col_name] = df[col_name].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Create mapping dictionary
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        logging.info(f"Label Encoding Mapping for {col_name}: {mapping}")

        return df, mapping

    except Exception as e:
        logging.error(f"Error occurred when encoding {col_name}: {e}")
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
