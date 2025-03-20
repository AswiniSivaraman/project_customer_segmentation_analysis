import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple
import os
import pickle


def apply_standard_scaling(train_df: pd.DataFrame, test_df: pd.DataFrame, numerical_cols: list, target_column: str, is_target_there: bool, pipeline_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Standard Scaling to numerical columns using training data to fit the scaler.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        numerical_cols (list): List of numerical columns to scale.
        target_column (str): Name of the target column (excluded from scaling).
        pipeline (str): Name of the pipeline (e.g., 'regression', 'classification', 'clustering').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test datasets (target column remains unchanged).
    """
    try:
        logging.info(f"Applying Standard Scaling for {pipeline_name} (excluding target column: {target_column})...")
        
        scaler_path = os.path.join("support", f"{pipeline_name}_standard_scaler.pkl")
        os.makedirs(pipeline_name, exist_ok=True)  # Ensure directory exists
        
        if is_target_there:
            features_to_scale = [col for col in numerical_cols if col != target_column]
        else:
            features_to_scale = numerical_cols
        
        if not os.path.exists(scaler_path):
            logging.info(f"Fitting StandardScaler for {pipeline_name} and saving to {scaler_path}")
            scaler = StandardScaler()
            train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
            test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
            
            with open(scaler_path, "wb") as file:
                pickle.dump(scaler, file)
        else:
            logging.info(f"Loading StandardScaler for {pipeline_name} from {scaler_path}")
            with open(scaler_path, "rb") as file:
                scaler = pickle.load(file)
            
            train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
            test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
        
        logging.info(f"Standard Scaling successfully applied to train & test data for {pipeline_name}.")
        
        return train_df, test_df
    
    except Exception as e:
        logging.error(f"Error applying Standard Scaling for {pipeline_name}: {e}")
        logging.exception("Full Exception Traceback:")
        raise e


def handle_imbalanced_data(train_df: pd.DataFrame, target_col: str, method: str = "smote") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handles class imbalance using either **SMOTE (Oversampling) or Random Undersampling** **ONLY on training data**.

    Args:
        train_df (pd.DataFrame): The training dataset.
        target_col (str): The target column for classification.
        method (str): Either `"smote"` for oversampling or `"undersampling"` for reducing majority class.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced training dataset (X_train, y_train)
    """
    try:
        logging.info(f"Started Applying Balancing technique --> {method} ...")

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        if method == "smote":
            logging.info("Applying SMOTE Oversampling on training data...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied. New class distribution:\n{y_resampled.value_counts()}")

        elif method == "undersampling":
            logging.info("Applying Random Undersampling on training data...")
            undersample = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersample.fit_resample(X_train, y_train)
            logging.info(f"Undersampling applied. New class distribution:\n{y_resampled.value_counts()}")

        else:
            raise ValueError("Invalid method. Choose 'smote' or 'undersampling'.")

        return X_resampled, y_resampled

    except Exception as e:
        logging.error(f"Error handling imbalanced data: {e}")
        logging.exception("Full Exception Traceback:")
        raise e
