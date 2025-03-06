import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple


def apply_standard_scaling(train_df: pd.DataFrame, test_df: pd.DataFrame, numerical_cols: list, target_column: str, is_target_there: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Standard Scaling to numerical columns using training data to fit the scaler.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        numerical_cols (list): List of numerical columns to scale.
        target_column (str): Name of the target column (excluded from scaling).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test datasets (target column remains unchanged).
    """
    try:
        logging.info(f"Applying Standard Scaling (excluding target column: {target_column})...")

        scaler = StandardScaler()
        
        if is_target_there:
            # Ensure target column is excluded
            features_to_scale = [col for col in numerical_cols if col != target_column]
        else:
            features_to_scale = numerical_cols
        
        # Fit scaler on train features and transform both train & test
        train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
        test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

        logging.info("Standard Scaling successfully applied to train & test data.")

        return train_df, test_df

    except Exception as e:
        logging.error(f"Error applying Standard Scaling: {e}")
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
