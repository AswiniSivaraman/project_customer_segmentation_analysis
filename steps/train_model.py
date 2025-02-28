import pandas as pd
import logging
from zenml import step
from src.model_training import train_regression_model, train_classification_mode, train_clustering_mode
import mlflow

@step
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
        with mlflow.start_run(run_name="Step_Train_Regression"):
            logging.info("Started Training Regression Models...")

            assert target_col in df.columns, f"Target column '{target_col}' not found in dataset"

            logging.info(f" Started to split the Training and Testing data")
            X_train = df.drop(columns=target_col)  # Drop target column from features
            y_train = df[target_col]  # Target column
            logging.info(f"Training and Testing data splitted successfully: {X_train.shape}, {y_train.shape}")

            logging.info(f"Started to train the models")
            model_paths = train_regression_model(X_train, y_train)
            logging.info(f"Successfully trained the models")

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
def train_classification(df: pd.DataFrame, target_col: str) -> dict:
    """
    ZenML Step to train multiple classification models.

    Args:
        df (pd.DataFrame): Training dataset.
        target_col (str): Target variable.

    Returns:
        dict: Paths of saved classification models.
    """
    try:
        with mlflow.start_run(run_name="Step_Train_Classification"):
            logging.info("Started Training Classification Models...")

            assert target_col in df.columns, f"Target column '{target_col}' not found in dataset"

            logging.info("Started to split the Training and Testing data")
            X_train = df.drop(columns=target_col)  # Drop target column from features
            y_train = df[target_col]  # Target column
            print(X_train.columns)
            print(y_train.head(1))
            logging.info(f"Training and Testing data splitted successfully: {X_train.shape}, {y_train.shape}")

            logging.info("Started to train the models")
            model_paths = train_classification_mode(X_train, y_train)
            logging.info("Successfully trained the models")

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
def train_clustering(df: pd.DataFrame) -> dict:
    """
    ZenML Step to train multiple clustering models.

    Args:
        df (pd.DataFrame): Training dataset.

    Returns:
        dict: Paths of saved clustering models.
    """
    try:
        with mlflow.start_run(run_name="Step_Train_Clustering"):
            logging.info("Started Training Clustering Models...")

            logging.info("Started to prepare the Training data")
            X_train = df  # No target column for clustering
            print(X_train.columns)
            logging.info(f"Training data prepared successfully: {X_train.shape}")

            logging.info("Started to train the models")
            model_paths = train_clustering_mode(X_train, None)
            logging.info("Successfully trained the models")

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
