import pandas as pd
import logging
from zenml import step
from src.find_best_model import select_best_model
from typing import Tuple

@step
def select_best_classification_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best classification model and return its name and metrics.

    Args:
    evaluation_df --> Dataframe contains evaluation values

    Return:
    tuple contains best model name and its metrices value
    """
    try:
        logging.info('Starting to find best classification model..')
        return select_best_model(evaluation_df, "classification")
    except Exception as e:
        logging.error(f'Error occured when finding the best regression model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise e 


@step
def select_best_regression_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best regression model and return its name and metrics.

    Args:
    evaluation_df --> Dataframe contains evaluation values

    Return:
    tuple contains best model name and its metrices value
    """
    try:
        logging.info('Starting to find best regression model..')
        return select_best_model(evaluation_df, "regression")
    except Exception as e:
        logging.error(f'Error occured when finding the best regression model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise e 


@step
def select_best_clustering_model(evaluation_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    ZenML step to select the best clustering model and return its name and metrics.

    Args:
    evaluation_df --> Dataframe contains evaluation values

    Return:
    tuple contains best model name and its metrices value
    """
    try:
        logging.info('Starting to find best clustering model..')
        return select_best_model(evaluation_df, "clustering")
    except Exception as e:
        logging.error(f'Error occured when finding the best clustering model in "steps"')
        logging.exception('Full Exception Traceback:')
        raise e    
