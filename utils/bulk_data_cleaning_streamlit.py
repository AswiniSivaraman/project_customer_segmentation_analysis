import logging
import os
import pandas as pd
import streamlit as st
import pickle

from src.data_preprocessing import clean_data
from src.data_feature_engineering import target_feature, new_feature_1, new_feature_2, new_feature_3

# Define complete lists for each pipeline type
clustering_features = [
    'month', 'day', 'order', 'country', 'session_id', 'page1_main_category',
    'page2_clothing_model', 'colour', 'location', 'model_photography', 'price',
    'price_2', 'page', 'purchase_completed', 'is_weekend', 'season',
    'total_clicks', 'max_page_reached'
]

regression_features = [
    'country', 'session_id', 'page1_main_category', 'page2_clothing_model',
    'colour', 'model_photography', 'price_2', 'page', 'is_weekend',
    'max_page_reached', 'cluster', 'price'
]

classification_features = [
    'order', 'page1_main_category', 'page2_clothing_model', 'location',
    'price', 'price_2', 'page', 'max_page_reached', 'cluster',
    'purchase_completed'
]

def encode_data(df, encoded_mapping, cat_cols=['page2_clothing_model', 'season', 'purchase_completed']):
    """
    Encodes the specified categorical columns in the DataFrame based on the provided encoded_mapping.
    """
    df_encoded = df.copy()
    for col in cat_cols:
        if col in df_encoded.columns and col in encoded_mapping:
            df_encoded[col] = df_encoded[col].apply(lambda x: encoded_mapping[col].get(x, x))
    return df_encoded

def process_bulk_data(uploaded_file, pipeline_name: str):
    """
    Process the uploaded CSV data and return a processed DataFrame for the given pipeline.
    
    For regression and classification, this function:
      1. Reads and cleans the data, and creates new features.
      2. Applies label encoding using the appropriate mapping.
      3. For regression/classification, predicts the "cluster" using the clustering model before scaling.
      4. Drops target columns (and the cluster column) prior to scaling.
      5. Reorders the DataFrame to match the scaler's fitted feature order and scales the data.
      6. Adds back the dropped columns.
      7. Checks for missing expected features (e.g. "is_weekend") and, if absent, adds them from the original data.
      8. Finally, selects the final feature set for that pipeline.
    
    For clustering, all features are scaled.
    """
    try:
        # Ingest Data
        df = pd.read_csv(uploaded_file)
        logging.info("Data successfully ingested from the uploaded file.")
        
        # Step 1: Clean Data
        df_clean_data = clean_data(df)
        logging.info("Data cleaning completed.")
        print("Data cleaning completed.")
        
        # Step 2: Create New Features
        df_clean_data = target_feature(df_clean_data, col1="page", col2="price", col3="order", feature_name="purchase_completed")
        logging.info("Target feature `purchase_completed` created.")
        print("Target feature `purchase_completed` created.")
        
        df_clean_data = new_feature_1(df_clean_data, col1="day", feature_name="is_weekend")
        logging.info("Time-based feature `is_weekend` created.")
        print("Time-based feature `is_weekend` created.")
        
        df_clean_data = new_feature_2(df_clean_data, col1="month", feature_name="season")
        logging.info("Season-based feature `season` created.")
        print("Season-based feature `season` created.")
        
        df_clean_data = new_feature_3(df_clean_data, col1="session_id", col2="order", transform_type="count", feature_name="total_clicks")
        logging.info("Session-based feature `total_clicks` created.")
        print("Session-based feature `total_clicks` created.")
        
        df_clean_data = new_feature_3(df_clean_data, col1="session_id", col2="page", transform_type="max", feature_name="max_page_reached")
        logging.info("Session-based feature `max_page_reached` created.")
        print("Session-based feature `max_page_reached` created.")
        
        # Step 3: Dynamic Label Encoding
        base_dir = os.path.dirname(__file__)
        mapping_file_path = os.path.join(base_dir, "..", "support", f"{pipeline_name}_encoded_mappings.pkl")
        try:
            with open(mapping_file_path, "rb") as f:
                encoded_mapping = pickle.load(f)
            logging.info(f"Loaded encoded mapping from {mapping_file_path}.")
            print(f"Loaded encoded mapping from {mapping_file_path}.")
        except Exception as e:
            logging.error("Error loading encoding mapping from " + mapping_file_path)
            logging.exception(e)
            st.error("An error occurred loading the encoding mapping file. Please check the logs.")
            return None
        
        df_encoded = encode_data(df_clean_data, encoded_mapping)
        logging.info("Data encoding completed using the mapping file.")
        print("Data encoding completed.")
        
        # Step 4: Load the appropriate scaler
        scaler_file_path = os.path.join(base_dir, "..", "support", f"{pipeline_name.lower()}_standard_scaler.pkl")
        try:
            with open(scaler_file_path, "rb") as f:
                scaler = pickle.load(f)
            logging.info(f"Loaded scaler from {scaler_file_path}.")
            print(f"Loaded scaler from {scaler_file_path}.")
        except Exception as e:
            logging.error("Error loading scaler from " + scaler_file_path)
            logging.exception(e)
            st.error("An error occurred loading the scaler file. Please check the logs.")
            return None
        
        # Helper: Reorder DataFrame columns to match scaler's fitted order (keeping only available ones).
        def reorder_df(df_to_scale, scaler_obj):
            if hasattr(scaler_obj, "feature_names_in_"):
                expected = list(scaler_obj.feature_names_in_)
                available = [feat for feat in expected if feat in df_to_scale.columns]
                return df_to_scale[available]
            return df_to_scale
        
        # For regression and classification, predict the "cluster" BEFORE scaling.
        if pipeline_name.lower() in ["regression", "classification"]:
            # Use the clustering scaler for proper ordering.
            clust_scaler_path = os.path.join(base_dir, "..", "support", "clustering_standard_scaler.pkl")
            with open(clust_scaler_path, "rb") as f:
                clust_scaler = pickle.load(f)
            df_for_cluster = reorder_df(df_encoded.copy(), clust_scaler)
            # Load clustering model
            clustering_model_path = os.path.join(base_dir, "..", "models", "clustering_models", "KMeans.pkl")
            with open(clustering_model_path, "rb") as f:
                clustering_model = pickle.load(f)
            cluster_pred = clustering_model.predict(df_for_cluster)
            df_encoded["cluster"] = cluster_pred
        
        # Step 5: Scaling the Data
        if pipeline_name.lower() == "regression":
            # Save target 'price' and 'cluster'
            price = df_encoded["price"]
            cluster_col = df_encoded["cluster"]
            # Drop target columns before scaling
            df_to_scale = df_encoded.drop(["price", "cluster"], axis=1)
            df_to_scale = reorder_df(df_to_scale, scaler)
            scaled_array = scaler.transform(df_to_scale)
            df_scaled = pd.DataFrame(scaled_array, columns=df_to_scale.columns, index=df_to_scale.index)
            # Add back the dropped columns
            df_scaled["price"] = price
            df_scaled["cluster"] = cluster_col
            # For any expected column (like 'is_weekend') missing from df_scaled, add from df_encoded.
            for col in regression_features:
                if col not in df_scaled.columns and col in df_encoded.columns:
                    df_scaled[col] = df_encoded[col]
            logging.info("Scaling applied for regression (excluding 'price' and 'cluster') with missing columns added.")
            print("Scaling applied for regression (excluding 'price' and 'cluster') with missing columns added.")
        elif pipeline_name.lower() == "classification":
            # Save target 'purchase_completed' and 'cluster'
            target = df_encoded["purchase_completed"]
            cluster_col = df_encoded["cluster"]
            # Drop target columns before scaling
            df_to_scale = df_encoded.drop(["purchase_completed", "cluster"], axis=1)
            df_to_scale = reorder_df(df_to_scale, scaler)
            scaled_array = scaler.transform(df_to_scale)
            df_scaled = pd.DataFrame(scaled_array, columns=df_to_scale.columns, index=df_to_scale.index)
            # Add back the dropped columns
            df_scaled["purchase_completed"] = target
            df_scaled["cluster"] = cluster_col
            logging.info("Scaling applied for classification (excluding 'purchase_completed' and 'cluster').")
            print("Scaling applied for classification (excluding 'purchase_completed' and 'cluster').")
        else:  # Clustering: scale all features.
            df_to_scale = reorder_df(df_encoded.copy(), scaler)
            scaled_array = scaler.transform(df_to_scale)
            df_scaled = pd.DataFrame(scaled_array, columns=df_to_scale.columns, index=df_to_scale.index)
            logging.info("Scaling applied for clustering (all features scaled).")
            print("Scaling applied for clustering (all features scaled).")
        
        # Step 6: Select the final features based on pipeline type.
        if pipeline_name.lower() == "clustering":
            selected_features = clustering_features
        elif pipeline_name.lower() == "regression":
            selected_features = regression_features
        elif pipeline_name.lower() == "classification":
            selected_features = classification_features
        else:
            selected_features = df_scaled.columns.tolist()  # fallback to all columns if pipeline not recognized
        
        # Ensure all expected features are present (if missing, add from original df_encoded)
        for col in selected_features:
            if col not in df_scaled.columns and col in df_encoded.columns:
                df_scaled[col] = df_encoded[col]
        
        df_final = df_scaled[selected_features].copy()
        logging.info("Final features selected based on pipeline type.")
        print("Final features selected based on pipeline type.")
        
        return df_final

    except Exception as e:
        logging.error("Error occurred in processing bulk data.")
        logging.exception(e)
        st.error("An error occurred during data processing. Please check the logs.")
        return None
