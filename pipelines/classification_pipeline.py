import logging
import pandas as pd
from zenml import pipeline
from zenml.client import Client
from steps.ingest_data import ingest_train_data, ingest_test_data
from steps.clean_data import preprocessing_data
from steps.feature_engineering import apply_feature_engineering
from steps.feature_selection import select_feature_classification
from steps.transforming_data import scale_features, balance_train_data
from steps.cluster_column import add_cluster_column
from steps.rfe_feature_selection import apply_rfe
from steps.train_model import train_classification
from steps.evaluate_model import evaluate_classification_step
from steps.best_model import select_best_classification_model
import mlflow

cat_cols = ['page2_clothing_model', 'season', 'purchase_completed']
cont_cols = ['month','day','order','country','session_id','page1_main_category','colour','location','model_photography','price','price_2','page','purchase_completed','is_weekend','total_clicks','max_page_reached']
columns = ['month', 'day', 'order', 'country', 'session_id', 'page1_main_category', 'page2_clothing_model', 'colour', 'location', 'model_photography', 'price', 'price_2', 'page', 'purchase_completed', 'is_weekend', 'season', 'total_clicks', 'max_page_reached']

@pipeline(enable_cache=False)
def classification_pipeline(train_data_path: str, test_data_path: str):
    """
    ZenML pipeline to train, evaluate, and select the best classification model.
    """
    try:
        logging.info("Starting Classification Pipeline...")
        # # Start MLflow Experiment Tracking
        # experiment_tracker = Client().active_stack.experiment_tracker
        # if experiment_tracker:
        #     logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        with mlflow.start_run(run_name='Classification Pipeline', nested=True) as run:
            run_id = run.info.run_id
            logging.info(f"MLflow run started, Active Run ID: {run_id}")

            # Step 1: Ingest Data
            df_train = ingest_train_data(train_data_path)
            df_test = ingest_test_data(test_data_path)
            print("Data fetched successfully!")

            # Step 2: Data Preprocess
            df_train_cleaned, df_test_cleaned = preprocessing_data(df_train, df_test, cat_cols, columns, "support", "classification")
            print("Data Preprocessed")

            # Step 3: Feature Selection
            selected_features = select_feature_classification(df_train_cleaned, continuous_cols=cont_cols, categorical_cols=cat_cols, target_col="purchase_completed")
            print("Feature selection completed")

            # Step 4: Scale & Balance Data
            train_scaled, test_scaled, feature_names = scale_features(df_train_cleaned, df_test_cleaned, numerical_cols=selected_features, target_column="purchase_completed", is_target_there=True, pipeline_type="classification")
            
            # Step 5: Add New Feature called "cluster"
            df_train_final, df_test_final = add_cluster_column(train_scaled, test_scaled, feature_names)
            print("New cluster column added")

            # Step 6: Apply RFE method to select the feature to train the model
            train_rfe, test_rfe, rfe_features = apply_rfe(df_train_final, df_test_final, target_column='purchase_completed', n_features=9, estimator='classification')
            print("RFE applied")
            
            # Step 7: Apply "smote" Method to Balance the Data
            X_train_balanced, y_train_balanced = balance_train_data(train_rfe, target_col="purchase_completed", method="smote")
            print("Data scaled and balanced")

            # Step 8: Train Model
            trained_classification_models = train_classification(X_train_balanced, y_train_balanced)
            print("Model trained")

            # Step 9: Evaluate Model
            classification_eval_df = evaluate_classification_step(test_rfe, target_column="purchase_completed", dependency=trained_classification_models) 
            print("Model evaluation completed")

            # Step 10: Select Best Model
            best_classification_model, classification_metrics = select_best_classification_model(classification_eval_df)
            print("Best model selected")

            # Log Best Model & Metrics
            logging.info(f"Best Classification Model: {best_classification_model}")
            logging.info(f"Classification Metrics: {classification_metrics}")

            return best_classification_model, classification_metrics

    except Exception as e:
        logging.error("Error occurred when running the classification pipeline")
        logging.exception("Full Exception Traceback:")

    finally:
        if mlflow.active_run():
            mlflow.end_run()
