import pandas as pd
import logging
from zenml import step
from src.find_best_model import select_best_model
from typing import Tuple
import mlflow

@step
def select_best_classification_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best classification model and return its name and metrics.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing evaluation values.

    Returns:
        Tuple[str, pd.DataFrame]: Best model name and its metrics.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        logging.info('Starting to find best classification model...')
        best_model_name, best_model_metrics = select_best_model(evaluation_df, "classification")
        return best_model_name, best_model_metrics
    except Exception as e:
        logging.error(f'Error occurred when finding the best classification model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise



@step
def select_best_regression_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best regression model and return its name and metrics.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing evaluation values.

    Returns:
        Tuple[str, pd.DataFrame]: Best model name and its metrics.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        logging.info('Starting to find best regression model...')
        best_model_name, best_model_metrics = select_best_model(evaluation_df, "regression")
        return best_model_name, best_model_metrics
    except Exception as e:
        logging.error(f'Error occurred when finding the best regression model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise


@step
def select_best_clustering_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best clustering model and return its name and metrics.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing evaluation values.

    Returns:
        Tuple[str, pd.DataFrame]: Best model name and its metrics.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
            
        logging.info('Starting to find best clustering model...')
        best_model_name, best_model_metrics = select_best_model(evaluation_df, "clustering")
        return best_model_name, best_model_metrics
    except Exception as e:
        logging.error(f'Error occurred when finding the best clustering model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise
  
