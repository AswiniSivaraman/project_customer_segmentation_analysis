import pandas as pd
import logging
from typing import Tuple
import mlflow
import os

def select_best_model(evaluation_df: pd.DataFrame, model_type: str) -> Tuple[str, pd.DataFrame]:
    """
    Selects the best model from the evaluation DataFrame and returns its name along with its metrics.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing model evaluation results.
        model_type (str): Type of model ('classification', 'regression', 'clustering').

    Returns:
        tuple:
            - str: The best model name.
            - pd.DataFrame: A single-row DataFrame with the best model's evaluation metrics.
    """
    try:
        logging.info(f"Selecting best model for {model_type}...")

        if evaluation_df.empty:
            logging.warning("Evaluation DataFrame is empty. No models to select from.")
            return None, None

        if model_type == "classification":
            # Select model with highest Accuracy
            best_model = evaluation_df["Accuracy"].idxmax()
        
        elif model_type == "regression":
            # Select model with lowest Mean Squared Error (MSE)
            best_model = evaluation_df["MSE"].idxmin()
        
        elif model_type == "clustering":
            # Select model with highest Silhouette Score
            best_model = evaluation_df["Silhouette Score"].idxmax()
        
        else:
            raise ValueError("Invalid model type. Choose 'classification', 'regression', or 'clustering'.")

        best_model_metrics = evaluation_df.loc[[best_model]]  # Extract its metrics

        # Save Best Model Path
        best_model_path = os.path.join("models", f"best_models", f"{model_type}_{best_model}.pkl")

        # Log Best Model in MLflow
        with mlflow.start_run(run_name=f"Best_{model_type}_Model"):
            mlflow.log_param("best_model", best_model)
            
            # Log Model Metrics
            for metric, value in best_model_metrics.items():
                mlflow.log_metric(f"best_{metric}", value)
            
            # Log Best Model as Artifact
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path)
            else:
                logging.warning(f"Model file not found: {best_model_path}")

        logging.info(f"Best {model_type} Model: {best_model}")
        logging.info(f"Best Model Metrics:\n{best_model_metrics}")

        return best_model, best_model_metrics

    except Exception as e:
        logging.error(f"Error selecting best model: {e}")
        raise e
