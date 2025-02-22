import pandas as pd
import logging
from zenml import step
from zenml import step, ArtifactConfig
from src.data_loading import load_train_data,load_test_data
from typing_extensions import Annotated

@step
def ingest_train_data(path: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="train_data")]:
    """
    ZenML step to load data from a given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in ingest_data.py file in steps')
        logging.info(f'Ingesting data from: {path}')

        df = load_train_data(path)
        
        logging.info(f'Data ingestion successful: {df.shape[0]} rows, {df.shape[1]} columns.')
        print(f'Data ingestion successful: {df.shape[0]} rows, {df.shape[1]} columns.')
        return df
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        logging.exception('Full Exception Traceback:')
        raise e


@step
def ingest_test_data(path: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="test_data")]:
    """
    ZenML step to load data from a given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in ingest_data.py file in steps')
        logging.info(f'Ingesting data from: {path}')

        df = load_test_data(path)
        
        logging.info(f'Data ingestion successful: {df.shape[0]} rows, {df.shape[1]} columns.')
        print(f'Data ingestion successful: {df.shape[0]} rows, {df.shape[1]} columns.')
        return df
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        logging.exception('Full Exception Traceback:')
        raise e