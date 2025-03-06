import os
import pickle
import logging
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, silhouette_score, davies_bouldin_score
import mlflow
import numpy as np

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
        logging.error(f'Error occurred when loading the model from the path --> {model_path}')
        logging.exception('Full Exception Traceback:')
        raise e


# Function to evaluate regression models
def evaluate_regression_models(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained regression models.
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
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[model_name] = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2_Score": r2}
                
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("R2_Score", r2)

        return results
    except Exception as e:
        logging.error(f'Error occurred when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e
    finally:
        if mlflow.active_run():
            mlflow.end_run()


# Function to evaluate classification models
def evaluate_classification_models(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained classification models.
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
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")
                roc_auc = roc_auc_score(y_test, y_pred, average="weighted")
                results[model_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc
                }
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("F1 Score", f1)
                    mlflow.log_metric("ROC AUC", roc_auc)

        return results
    except Exception as e:
        logging.error(f'Error occurred when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e
    finally:
        if mlflow.active_run():
            mlflow.end_run()

# Function to evaluate clustering models
def evaluate_clustering_models(X_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Load and evaluate all trained clustering models.
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
                silhouette = silhouette_score(X_test, labels)
                davies_bouldin = davies_bouldin_score(X_test, labels)
                results[model_name] = {"Silhouette Score": silhouette, "Davies-Bouldin Index": davies_bouldin}
                with mlflow.start_run(run_name=f"Eval_{model_name}"):
                    mlflow.log_metric("Silhouette_Score", silhouette)
                    mlflow.log_metric("Davies_Bouldin_Index", davies_bouldin)

        return results
    except Exception as e:
        logging.error(f'Error occurred when evaluating the {model_name} model')
        logging.exception('Full Exception Traceback:')
        raise e
    finally:
        if mlflow.active_run():
            mlflow.end_run()
