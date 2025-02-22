import os
import pickle
import logging
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,mean_squared_error, mean_absolute_error, r2_score,silhouette_score
import mlflow

# Define function to load a saved model
def load_model(model_name: str, model_type: str):
    """
    Load a trained model from disk.

    Args:
        model_name (str): Name of the model to load.
        model_type (str): Type of model ('regression', 'classification', 'clustering').

    Returns:
        Trained model instance.
    """

    try:
        model_path = os.path.join("models", f"{model_type}_models", f"{model_name}.pkl")

        if not os.path.exists(model_path):
            logging.error(f"Model file {model_path} not found!")
            return None

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        return model
    
    except Exception as e:
        logging.error(f'Error occured when load the model from the path --> {model_path}')
        logging.exception('Full Exception Traceback:')
        raise e 

# Function to evaluate regression models
def evaluate_regression_models(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained regression models.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Model names and their evaluation metrics.
    """
    try:
        results = {}

        regression_models = [
            "LinearRegression", "PolynomialRegression", "RidgeRegression",
            "LassoRegression", "ElasticNet", "SVR", "DecisionTreeRegressor",
            "RandomForestRegressor", "GradientBoostingRegressor", "XGBoostRegressor", "MLPRegressor"
        ]

        for model_name in regression_models:
            model = load_model(model_name, "regression")

            if model:
                y_pred = model.predict(X_test)

                # Compute metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results[model_name] = {"MSE": mse, "MAE": mae, "R² Score": r2}
                logging.info(f"{model_name}: MSE = {mse:.3f}, R² = {r2:.3f}")
                
                # Log Metrics in MLflow
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("R2_Score", r2)

        return results
    
    except Exception as e:
        logging.error(f'Error occured when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e

# Function to evaluate classification models
def evaluate_classification_models(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained classification models.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Model names and their evaluation metrics.
    """
    try:
        results = {}

        classification_models = [
            "LogisticRegression", "SVC", "DecisionTreeClassifier",
            "RandomForestClassifier", "GradientBoostingClassifier",
            "KNeighborsClassifier", "GaussianNB", "MLPClassifier", "XGBoostClassifier"
        ]

        for model_name in classification_models:
            model = load_model(model_name, "classification")

            if model:
                y_pred = model.predict(X_test)

                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                results[model_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                }

                logging.info(f"{model_name}: Accuracy = {accuracy:.3f}, F1 Score = {f1:.3f}")

                # Log Metrics in MLflow
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("F1 Score", f1)

        return results
    except Exception as e:
        logging.error(f'Error occured when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e

# Function to evaluate clustering models
def evaluate_clustering_models(X_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained clustering models using Silhouette Score.

    Args:
        X_test (pd.DataFrame): Test features.

    Returns:
        dict: Model names and their evaluation metrics.
    """
    try:
        results = {}

        clustering_models = [
            "KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture"
        ]

        for model_name in clustering_models:
            model = load_model(model_name, "clustering")

            if model:
                labels = model.predict(X_test) if hasattr(model, "predict") else model.fit_predict(X_test)

                # Compute silhouette score
                score = silhouette_score(X_test, labels)
                results[model_name] = {"Silhouette Score": score}

                logging.info(f"{model_name}: Silhouette Score = {score:.3f}")

                # Log Metrics in MLflow
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("Silhouette_Score", score)

        return results
    except Exception as e:
        logging.error(f'Error occured when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e
