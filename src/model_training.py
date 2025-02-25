import pandas as pd
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from typing import Tuple
from utils.helper_func import train_any_model
import mlflow

regression_type_models = {
    "LinearRegression": LinearRegression(),
    "PolynomialRegression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "RidgeRegression": Ridge(alpha=1.0),
    "LassoRegression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "SVR": SVR(kernel="rbf"),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoostRegressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}

def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains multiple regression models using the dictionary `regression_type_models`.
    
    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        dict: Model names and their corresponding saved file paths.
    """
    try:
        model_paths = {}
        with mlflow.start_run(run_name="Train_Regression_Models"):
            for name, model in regression_type_models.items():
                logging.info(f'training {name} model')
                trained_reg_model, path = train_any_model(model, X_train, y_train, model_name=name, model_type="regression")
                
                # Log Model Path in MLflow
                mlflow.log_param(f"{name}_path", path)
                
                logging.info(f"{name} saved at: {path}")
                print(f"{name} saved at: {path}")
                model_paths[name] = path

        return model_paths
    except Exception as e:
        logging.error(f'Error occured when training the model {model}')
        logging.exception('Full Exception Traceback: ')
        raise e


classification_type_models = {
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(kernel="rbf", probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB(),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoostClassifier": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", random_state=42)
}

def train_classification_mode(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains multiple classification models using the dictionary `classification_type_models`.
    
    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        dict: Model names and their corresponding saved file paths.
    """
    try:
        model_paths = {}
        with mlflow.start_run(run_name="Train_Classification_Models"):
            for name, model in classification_type_models.items():
                logging.info(f'training {name} model')
                trained_reg_model, path = train_any_model(model, X_train, y_train, model_name=name, model_type="classification")

                # Log Model Path in MLflow
                mlflow.log_param(f"{name}_path", path)
                
                logging.info(f"{name} saved at: {path}")
                print(f"{name} saved at: {path}")
                model_paths[name] = path

        return model_paths
    except Exception as e:
        logging.error(f'Error occured when training the model {model}')
        logging.exception('Full Exception Traceback: ')
        raise e


clustering_type_models = {
    "KMeans": KMeans(n_clusters=5, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=5),
    "GaussianMixture": GaussianMixture(n_components=5, random_state=42)
}

def train_clustering_mode(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains multiple clustering models using the dictionary `clustering_type_models`.
    
    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        dict: Model names and their corresponding saved file paths.
    """

    try:
        model_paths = {}
        with mlflow.start_run(run_name="Train_Clustering_Models"):
            for name, model in clustering_type_models.items():
                logging.info(f'training {name} model')
                trained_reg_model, path = train_any_model(model, X_train, y_train, model_name=name, model_type="clustering")
                
                # Log Model Path in MLflow
                mlflow.log_param(f"{name}_path", path)

                logging.info(f"{name} saved at: {path}")
                print(f"{name} saved at: {path}")
                model_paths[name] = path

        return model_paths
    except Exception as e:
        logging.error(f'Error occured when training the model {model}')
        logging.exception('Full Exception Traceback: ')
        raise e
