import pandas as pd
import logging
from zenml import step
from src.model_evaluation import (
    evaluate_regression_models,
    evaluate_classification_models,
    evaluate_clustering_models
)

@step
def evaluate_regression_step(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained regression models and return results as a DataFrame.

    Args:
        X_test (pd.DataFrame): Test dataset features.
        y_test (pd.Series): Test dataset target values.

    Returns:
        pd.DataFrame: DataFrame containing MSE, MAE, and R² Score for each regression model.
    """
    try:
        logging.info("Starting regression model evaluation...")
        results = evaluate_regression_models(X_test, y_test)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info(f"Regression model evaluation completed.")
        return results_df

    except Exception as e:
        logging.error(f"Error in regression model evaluation: {e}")
        raise e


@step
def evaluate_classification_step(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained classification models and return results as a DataFrame.

    Args:
        X_test (pd.DataFrame): Test dataset features.
        y_test (pd.Series): Test dataset target values.

    Returns:
        pd.DataFrame: DataFrame containing Accuracy, Precision, Recall, and F1 Score for each classification model.
    """
    try:
        logging.info("Starting classification model evaluation...")
        results = evaluate_classification_models(X_test, y_test)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info(f"Classification model evaluation completed.")
        return results_df

    except Exception as e:
        logging.error(f"Error in classification model evaluation: {e}")
        raise e


@step
def evaluate_clustering_step(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained clustering models and return results as a DataFrame.

    Args:
        X_test (pd.DataFrame): Test dataset features.

    Returns:
        pd.DataFrame: DataFrame containing Silhouette Score for each clustering model.
    """
    try:
        logging.info("Starting clustering model evaluation...")
        results = evaluate_clustering_models(X_test)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info(f"Clustering model evaluation completed.")
        return results_df

    except Exception as e:
        logging.error(f"Error in clustering model evaluation: {e}")
        raise e
