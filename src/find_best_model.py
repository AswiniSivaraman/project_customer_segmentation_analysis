import pandas as pd
import logging
from typing import Tuple
import mlflow
import os

def select_best_model(evaluation_df: pd.DataFrame, model_type: str) -> Tuple[str, dict]:
    """
    Selects the best model from the evaluation DataFrame, logs it to MLflow,
    and stores it in a CSV file inside 'models/best_models/' while ensuring unique model names.

    Args:
        evaluation_df (pd.DataFrame): DataFrame containing model evaluation results.
        model_type (str): Type of model ('classification', 'regression', 'clustering').

    Returns:
        tuple:
            - str: The best model name.
            - dict: A dictionary with the best model's evaluation metrics.
    """
    try:
        # Define the correct path inside "models/best_models/"
        best_models_dir = "models/best_models"
        best_models_file = os.path.join(best_models_dir, "best_models_file.csv")

        # Ensure the directory exists
        os.makedirs(best_models_dir, exist_ok=True)

        logging.info(f"Selecting best model for {model_type}...")

        if evaluation_df.empty:
            logging.warning("Evaluation DataFrame is empty. No models to select from.")
            return None, None

        # Ensure model_type is valid
        valid_model_types = ["classification", "regression", "clustering"]
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model type '{model_type}'. Choose from {valid_model_types}.")

        # Determine the best model based on evaluation metric
        if model_type == "classification":
            best_model_index = evaluation_df["Accuracy"].idxmax()
        elif model_type == "regression":
            best_model_index = evaluation_df["MSE"].idxmin()  # Finds row index with lowest MSE
        elif model_type == "clustering":
            best_model_index = evaluation_df["Silhouette Score"].idxmax()

        # Retrieve the model name from the 'Model' column
        best_model_name = evaluation_df.loc[best_model_index, "Model"]

        # Convert best_model_name to string to avoid indexing issues
        best_model_name = str(best_model_name)

        # Get model's performance metrics as a dictionary
        best_model_metrics = evaluation_df.loc[best_model_index].to_dict()

        print(f"Model Type: {model_type}")
        print(f"Best Model: {best_model_name}")
        print(f"Best Model Metrics: {best_model_metrics}")

        # Ensure the CSV file exists or create an empty DataFrame
        if os.path.exists(best_models_file):
            best_models_df = pd.read_csv(best_models_file)
        else:
            best_models_df = pd.DataFrame(columns=["Model Type", "Best Model"])

        # Remove any previous entry of the same model type
        best_models_df = best_models_df[best_models_df["Model Type"] != model_type]

        # Append the new best model while maintaining only 3 unique entries
        new_entry = pd.DataFrame({"Model Type": [model_type], "Best Model": [best_model_name]})
        best_models_df = pd.concat([best_models_df, new_entry], ignore_index=True)

        # Save the updated DataFrame back to the CSV file in 'models/best_models/'
        best_models_df.to_csv(best_models_file, index=False)

        logging.info(f"Best model saved in {best_models_file}")
        print(f"Best model saved in {best_models_file}")

        # Start an MLflow run to log the best model
        with mlflow.start_run(run_name=f"Best_{model_type}_Model"):
            mlflow.log_param("best_model", best_model_name)  # Log best model name

            # Log appropriate metrics based on model type
            if model_type == "regression":
                regression_metrics = ["MSE", "MAE", "RMSE", "R2_Score"]
                for metric in regression_metrics:
                    if metric in best_model_metrics and isinstance(best_model_metrics[metric], (int, float)):
                        mlflow.log_metric(metric, best_model_metrics[metric])

            elif model_type == "classification":
                classification_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
                for metric in classification_metrics:
                    if metric in best_model_metrics and isinstance(best_model_metrics[metric], (int, float)):
                        mlflow.log_metric(metric, best_model_metrics[metric])

            elif model_type == "clustering":
                clustering_metrics = ["Silhouette Score", "Davies-Bouldin Index"]
                for metric in clustering_metrics:
                    if metric in best_model_metrics and isinstance(best_model_metrics[metric], (int, float)):
                        mlflow.log_metric(metric, best_model_metrics[metric])


            # Log the CSV file as an artifact
            if os.path.exists(best_models_file):
                mlflow.log_artifact(best_models_file)
            else:
                logging.warning(f"CSV file not found: {best_models_file}. It may not have been saved yet.")

        logging.info(f"Best {model_type} Model: {best_model_name}")
        logging.info(f"Best Model Metrics:\n{best_model_metrics}")

        return best_model_name, best_model_metrics

    except Exception as e:
        logging.error(f"Error selecting best model: {e}")
        raise e
    
    finally:
        if mlflow.active_run():
            mlflow.end_run()
