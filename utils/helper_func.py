import pandas as pd
import logging
import os
import pickle
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin
from typing import Tuple, Union
import mlflow

# note --> after developing the model, check whether to return the model or the model_filename

def train_any_model(
    model: Union[RegressorMixin, ClassifierMixin, ClusterMixin], 
    X_train: pd.DataFrame, 
    y_train: Union[pd.Series, None],  # y_train can be None for Clustering
    model_name: str, 
    model_type: str
) -> Tuple[Union[RegressorMixin, ClassifierMixin, ClusterMixin], str]:
    """
    Train any type of model (Regression, Classification, Clustering) and save it.

    Args:
        model (RegressorMixin | ClassifierMixin | ClusterMixin): Model instance to be trained.
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series | None): Target variable for training (None for Clustering).
        model_name (str): Name of the model.
        model_type (str): Model category ('regression', 'classification', 'clustering').

    Returns:
        Tuple:
            - Trained model instance.
            - Path to the saved model file.
    """
    try:
        logging.info(f"Training {model_name}...")

        # Start MLflow Run
        with mlflow.start_run(run_name=f"{model_type}_{model_name}"):

            # Train the model (handle clustering separately as it doesn't need y_train)
            if model_type == "clustering":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

            # Define directory paths
            base_dir = "models"
            model_dir = os.path.join(base_dir, f"{model_type}_models")
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_filename = os.path.join(model_dir, f"{model_name}.pkl")
            with open(model_filename, "wb") as file:
                pickle.dump(model, file)

            logging.info(f"{model_name} successfully saved at: {model_filename}")

            # Log Model Parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", model_name)

            # Log Model Artifact (Saving Model to MLflow)
            mlflow.log_artifact(model_filename)

            return model, model_filename

    except Exception as e:
        logging.error(f"Error in training {model_name}.")
        logging.exception('Full Exception Traceback:')
        raise e




