import pandas as pd
import logging
from zenml import ArtifactConfig
from typing_extensions import Annotated

def load_train_data(path: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="train_data")]:    
    """
    Load the train data from the given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in load train data function in src/data_loading path')
        logging.info(f'Loading data from the path --> {path}')
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f'Error loading data from the path --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e
    

def load_test_data(path: str) -> Annotated[pd.DataFrame, ArtifactConfig(name="test_data")]:
    """
    Load the test data from the given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in load test data function in src/data_loading path')
        logging.info(f'Loading data from the path --> {path}')
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f'Error loading data from the path --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e
