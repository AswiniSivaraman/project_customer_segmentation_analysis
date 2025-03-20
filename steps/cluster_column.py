import pandas as pd
import pickle
import numpy as np
from typing import Tuple
from zenml import step

@step
def add_cluster_column(train_data: pd.DataFrame, test_data: pd.DataFrame, feature_columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the clustering model and adds a new column with cluster predictions for both training and testing data.
    
    If the clustering model has a 'predict' method (e.g. KMeans), it is used directly.
    For models like AgglomerativeClustering that lack 'predict', the training labels (labels_) are used
    to compute centroids and test samples are assigned to the nearest centroid.
    
    Args:
        train_data: The training DataFrame.
        test_data: The testing DataFrame.
        feature_columns: The list of columns used for clustering prediction.
    
    Returns:
        A tuple of DataFrames (updated_train_data, updated_test_data) with an added 'cluster' column.
    """
    # Load the CSV file that contains the best model information
    best_models_csv = "models/best_models/best_models_file.csv"
    best_models_df = pd.read_csv(best_models_csv)

    # Retrieve the name of the best clustering model
    clust_model_name = best_models_df.loc[best_models_df['Model Type'] == 'clustering', 'Best Model'].values[0]

    # Build the path to the clustering model file
    clustering_model_path = f"models/clustering_models/KMeans.pkl"
    
    # Load the clustering model from the .pkl file
    with open(clustering_model_path, "rb") as f:
        clustering_model = pickle.load(f)

    # Create copies of the input data to avoid modifying original DataFrames
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Check if the model supports predict
    if hasattr(clustering_model, "predict"):
        # For models like KMeans that have a predict method
        train_cluster_labels = clustering_model.predict(train_data[feature_columns])
        test_cluster_labels = clustering_model.predict(test_data[feature_columns])
    else:
        # For models like AgglomerativeClustering that don't support predict
        # Use the labels from the training phase
        if hasattr(clustering_model, "labels_"):
            train_cluster_labels = clustering_model.labels_
        else:
            # As a fallback, compute labels using fit_predict on train data
            train_cluster_labels = clustering_model.fit_predict(train_data[feature_columns])
        
        # Compute centroids from the training data
        centroids = train_data[feature_columns].groupby(train_cluster_labels).mean()

        # Define a helper function to assign the nearest centroid as the cluster
        def assign_cluster(row):
            distances = np.linalg.norm(centroids.values - row.values, axis=1)
            return centroids.index[np.argmin(distances)]
        
        # Apply the function to assign clusters to test data
        test_cluster_labels = test_data[feature_columns].apply(assign_cluster, axis=1)

    # Add the cluster predictions as a new column in each DataFrame
    train_data["cluster"] = train_cluster_labels
    test_data["cluster"] = test_cluster_labels

    print(f"Columns after adding cluster: {train_data.columns}")
    return train_data, test_data
