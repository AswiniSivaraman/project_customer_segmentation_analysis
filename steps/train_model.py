import pandas as pd
import logging
from zenml import step
from src.model_training import train_regression_model, train_classification_model, train_clustering_model
import mlflow
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_regression(df: pd.DataFrame, target_col: str) -> dict:
    """
    ZenML Step to train multiple regression models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable.

    Returns:
        dict: Paths of saved regression models.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        with mlflow.start_run(run_name="Step Train Regression"):
            logging.info("Started Training Regression Models...")

            assert target_col in df.columns, f"Target column '{target_col}' not found in dataset"

            logging.info(f" Started to split the Training and Testing data")
            X_train = df.drop(columns=target_col)  # Drop target column from features
            y_train = df[target_col]  # Target column
            logging.info(f"Training and Testing data splitted successfully: {X_train.shape}, {y_train.shape}")

            logging.info(f"Started to train the models")
            model_paths = train_regression_model(X_train, y_train)
            logging.info(f"Successfully trained the models")

            logging.info(f"Regression models trained and saved: {model_paths}")
            return model_paths

    except Exception as e:
        logging.error(f"Error in Regression Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e
    
    finally:
        if mlflow.active_run():
            mlflow.end_run()


@step(experiment_tracker=experiment_tracker.name)
def train_classification(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    ZenML Step to train multiple classification models.

    Args:
        df (pd.DataFrame): Training dataset.
        target_col (str): Target variable.

    Returns:
        dict: Paths of saved classification models.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        with mlflow.start_run(run_name="Step Train Classification"):
            logging.info("Started Training Classification Models...")
            logging.info(f"Data split completed: Features shape {X_train.shape}, Target shape {y_train.shape}")

            # Train the classification models
            logging.info("Starting the training of classification models...")
            model_paths = train_classification_model(X_train, y_train)
            logging.info(f"Successfully trained the classification models: {model_paths}")

            # Simply return the model paths
            return model_paths

    except Exception as e:
        logging.error(f"Error in Classification Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e

    finally:
        if mlflow.active_run():
            mlflow.end_run()


@step(experiment_tracker=experiment_tracker.name)
def train_clustering(df: pd.DataFrame) -> dict:
    """
    ZenML Step to train multiple clustering models.

    Args:
        df (pd.DataFrame): Dataset for clustering.

    Returns:
        dict: Paths of saved clustering models.
    """
    try:
        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        with mlflow.start_run(run_name="Step Train Clustering"):
            logging.info("Started Training Clustering Models...")

            # Prepare the data for clustering
            logging.info("Preparing the data for clustering...")
            logging.info(f"Dataset shape: {df.shape}, Columns: {list(df.columns)}")

            # Train the clustering models
            logging.info("Starting the training of clustering models...")
            model_paths = train_clustering_model(df)
            logging.info(f"Successfully trained the clustering models: {model_paths}")

            # Simply return the model paths
            return model_paths

    except Exception as e:
        logging.error(f"Error in Clustering Model Training: {e}")
        logging.exception('Full Exception Traceback:')
        raise e

    finally:
        if mlflow.active_run():
            mlflow.end_run()

