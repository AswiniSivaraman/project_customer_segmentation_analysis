import pandas as pd
import logging
from zenml import step
from src.model_evaluation import (
    evaluate_regression_models,
    evaluate_classification_models,
    evaluate_clustering_models
)
import mlflow
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_regression_step(df: pd.DataFrame, target_column: str, dependency: dict) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained regression models and return results as a DataFrame.

    Args:
        X_test (pd.DataFrame): Test dataset features.
        y_test (pd.Series): Test dataset target values.

    Returns:
        pd.DataFrame: DataFrame containing MSE, MAE, and R² Score for each regression model.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
            
        X_test = df.drop(columns=target_column)
        y_test = df[target_column]
        logging.info("Starting regression model evaluation...")
        results = evaluate_regression_models(X_test, y_test)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info(f"Regression model evaluation completed.")
        print(results_df)
        return results_df

    except Exception as e:
        logging.error(f"Error in regression model evaluation: {e}")
        mlflow.end_run()
        raise e


@step(experiment_tracker=experiment_tracker.name)
def evaluate_classification_step(df: pd.DataFrame, target_column: str, dependency: dict) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained classification models and return results as a DataFrame.

    Args:
        df (pd.DataFrame): Test dataset including features and target column.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: DataFrame containing Accuracy, Precision, Recall, and F1 Score for each classification model.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        
        X_test = df.drop(columns=target_column)
        y_test = df[target_column]
        logging.info("Starting classification model evaluation...")
        results = evaluate_classification_models(X_test, y_test)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info("Classification model evaluation completed.")
        return results_df

    except Exception as e:
        logging.error(f"Error in classification model evaluation: {e}")
        mlflow.end_run()
        raise e



@step(experiment_tracker=experiment_tracker.name)
def evaluate_clustering_step(df: pd.DataFrame, dependency: dict) -> pd.DataFrame:
    """
    ZenML Step to evaluate all trained clustering models and return results as a DataFrame.

    Args:
        df (pd.DataFrame): Test dataset including features.

    Returns:
        pd.DataFrame: DataFrame containing Silhouette Score for each clustering model.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
        
        logging.info("Starting clustering model evaluation...")
        results = evaluate_clustering_models(df)

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Model"}, inplace=True)

        logging.info("Clustering model evaluation completed.")
        return results_df

    except Exception as e:
        logging.error(f"Error in clustering model evaluation: {e}")
        mlflow.end_run()
        raise e

