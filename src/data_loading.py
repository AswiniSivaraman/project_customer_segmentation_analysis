import pandas as pd
import logging

def load_train_data(path: str) -> pd.DataFrame:    
    """
    Load the train data from the given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in load_train_data function in src/data_loading path')
        logging.info(f'Loading data from the path --> {path}')
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f'Error loading data from the path --> {e}')
        logging.exception('Full Exception Traceback:')
        raise e
    

def load_test_data(path: str) -> pd.DataFrame:
    """
    Load the test data from the given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        print('I am in load_test_data function in src/data_loading path')
        logging.info(f'Loading data from the path --> {path}')
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f'Error loading data from the path --> {e}')
        logging.exception('Full Exception Traceback:')
        raise e
