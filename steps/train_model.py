import pandas as pd
import logging
from zenml import step
from src.model_training import train_regression_model, train_classification_mode, train_clustering_mode
import mlflow

@step
def train_regression(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    ZenML Step to train multiple regression models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable.

    Returns:
        dict: Paths of saved regression models.
    """
    try:
        with mlflow.start_run(run_name="Step_Train_Regression"):
            logging.info("Started Training Regression Models...")
            model_paths = train_regression_model(X_train, y_train)

            # Log total models trained
            mlflow.log_param("total_models_trained", len(model_paths))

            # Log model paths
            for model_name, path in model_paths.items():
                mlflow.log_param(f"{model_name}_path", path)

            logging.info(f"Regression models trained and saved: {model_paths}")
            return model_paths

    except Exception as e:
        logging.error(f"Error in Regression Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e


@step
def train_classification(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    ZenML Step to train multiple classification models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable.

    Returns:
        dict: Paths of saved classification models.
    """
    try:
        with mlflow.start_run(run_name="Step_Train_Classification"):
            logging.info("Started Training Classification Models...")
            model_paths = train_classification_mode(X_train, y_train)

            # Log total models trained
            mlflow.log_param("total_models_trained", len(model_paths))

            # Log model paths
            for model_name, path in model_paths.items():
                mlflow.log_param(f"{model_name}_path", path)

            logging.info(f"Classification models trained and saved: {model_paths}")
            return model_paths

    except Exception as e:
        logging.error(f"Error in Classification Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e


@step
def train_clustering(X_train: pd.DataFrame) -> dict:
    """
    ZenML Step to train multiple clustering models.

    Args:
        X_train (pd.DataFrame): Training features (no target needed for clustering).

    Returns:
        dict: Paths of saved clustering models.
    """
    try:
        with mlflow.start_run(run_name="Step_Train_Clustering"):
            logging.info("Started Training Clustering Models...")
            model_paths = train_clustering_mode(X_train, None)  # No target column needed for clustering
            
            # Log total models trained
            mlflow.log_param("total_models_trained", len(model_paths))

            # Log model paths
            for model_name, path in model_paths.items():
                mlflow.log_param(f"{model_name}_path", path)

            logging.info(f"Clustering models trained and saved: {model_paths}")
            return model_paths

    except Exception as e:
        logging.error(f"Error in Clustering Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e
