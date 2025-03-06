import pandas as pd
import logging
from zenml import step
from src.data_feature_selection import (
    correlation,
    feature_selection_classification,
    feature_selection_regression,
    feature_selection_clustering
)
from typing import Tuple

@step
def select_feature_regression(df: pd.DataFrame, continuous_cols: list, categorical_cols: list, target_col: str) -> list:
    """
    ZenML Step to perform feature selection for regression models.

    Args:
        df (pd.DataFrame): The input dataset.
        continuous_cols (list): List of numerical feature columns.
        categorical_cols (list): List of categorical feature columns.
        target_col (str): The target column for regression.

    Returns:
        list: Selected features for regression.
    """
    try:
        logging.info("Starting feature selection for regression...")
        selected_features = feature_selection_regression(df, continuous_cols, categorical_cols, target_col)
        logging.info(f"Selected Features for Regression: {selected_features}")

        return selected_features

    except Exception as e:
        logging.error(f"Error in feature selection for regression: {e}")
        logging.exception("Full Exception Traceback:")
        raise e


@step
def select_feature_classification(df: pd.DataFrame, continuous_cols: list, categorical_cols: list, target_col: str) -> list:
    """
    ZenML Step to perform feature selection for classification models.

    Args:
        df (pd.DataFrame): The input dataset.
        continuous_cols (list): List of numerical feature columns.
        categorical_cols (list): List of categorical feature columns.
        target_col (str): The target column for classification.

    Returns:
        list: Selected features for classification.
    """
    try:
        logging.info("Starting feature selection for classification...")
        selected_features = feature_selection_classification(df, continuous_cols, categorical_cols, target_col)
        logging.info(f"Selected Features for Classification: {selected_features}")

        return selected_features

    except Exception as e:
        logging.error(f"Error in feature selection for classification: {e}")
        logging.exception("Full Exception Traceback:")
        raise e


@step
def select_feature_clustering(df: pd.DataFrame, variance_threshold: float = 0.01) -> Tuple[pd.DataFrame, list]:
    """
    ZenML Step to perform feature selection for clustering models using Variance Threshold.

    Args:
        df (pd.DataFrame): The input dataset.
        variance_threshold (float): Minimum variance a feature must have to be kept.

    Returns:
        pd.DataFrame: Dataset with selected features for clustering.
    """
    try:
        logging.info("Starting feature selection for clustering using Variance Threshold...")
        
        # Apply Variance Threshold
        df_selected = feature_selection_clustering(df, variance_threshold)
        selected_columns = df_selected.columns.tolist()
        logging.info(f"Selected Features for Clustering: {df_selected.columns.tolist()}")
        
        return df_selected, selected_columns

    except Exception as e:
        logging.error(f"Error in feature selection for clustering: {e}")
        logging.exception("Full Exception Traceback:")
        raise e