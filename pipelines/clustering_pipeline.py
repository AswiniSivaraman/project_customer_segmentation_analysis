import logging
import pandas as pd
from zenml import pipeline
from zenml.client import Client
from steps.ingest_data import ingest_train_data, ingest_test_data
from steps.clean_data import preprocessing_data
from steps.feature_engineering import apply_feature_engineering
from steps.feature_selection import select_feature_clustering
from steps.transforming_data import scale_features
from steps.train_model import train_clustering
from steps.evaluate_model import evaluate_clustering_step
from steps.best_model import select_best_clustering_model
import mlflow

cat_cols = ['page2_clothing_model', 'season', 'purchase_completed']
cont_cols = ['month','day','order','country','session_id','page1_main_category','colour','location','model_photography','price','price_2','page','purchase_completed','is_weekend','total_clicks','max_page_reached']
columns = ['month', 'day', 'order', 'country', 'session_id', 'page1_main_category', 'page2_clothing_model', 'colour', 'location', 'model_photography', 'price', 'price_2', 'page', 'purchase_completed', 'is_weekend', 'season', 'total_clicks', 'max_page_reached']

@pipeline(enable_cache=False)
def clustering_pipeline(train_data_path: str, test_data_path: str):
    """
    ZenML pipeline to train, evaluate, and select the best clustering model.
    """
    try:
        logging.info("Starting Clustering Pipeline...")
        # # Start MLflow Experiment Tracking
        # experiment_tracker = Client().active_stack.experiment_tracker
        # if experiment_tracker:
        #     logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        if mlflow.active_run():
            logging.info(f"Active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        with mlflow.start_run(run_name='Clustering Pipeline', nested=True) as run:
            run_id = run.info.run_id
            logging.info(f"MLflow run started, Active Run ID: {run_id}")

            # Step 1: Ingest Data
            df_train = ingest_train_data(train_data_path)
            df_test = ingest_test_data(test_data_path)
            print("Data fetched successfully!")

            # Step 2: Data Preprocess
            df_train_cleaned, df_test_cleaned = preprocessing_data(df_train, df_test, cat_cols, columns, "support")
            print("Data Preprocessed")

            # Step 3: Feature Selection
            df_selected_train, train_selected_columns = select_feature_clustering(df_train_cleaned)
            df_selected_test, test_selected_columns = select_feature_clustering(df_test_cleaned)
            print("Feature selection completed")

            # Step 4: Scale Data
            train_scaled, test_scaled = scale_features(df_selected_train, df_selected_test, numerical_cols=train_selected_columns, target_column="None", is_target_there=False)
            print("Data scaled")

            # Step 5: Train Model
            trained_clustering_models = train_clustering(train_scaled)
            print("Model trained")

            # Step 6: Evaluate Model
            clustering_eval_df = evaluate_clustering_step(test_scaled, dependency=trained_clustering_models)
            print("Model evaluation completed")

            # Step 7: Select Best Model
            best_clustering_model, clustering_metrics = select_best_clustering_model(clustering_eval_df)
            print("Best model selected")

            # Log Best Model & Metrics
            logging.info(f"Best Clustering Model: {best_clustering_model}")
            logging.info(f"Clustering Metrics: {clustering_metrics}")

            return best_clustering_model, clustering_metrics

    except Exception as e:
        logging.error("Error occurred when running the clustering pipeline")
        logging.exception("Full Exception Traceback:")

    finally:
        if mlflow.active_run():
            mlflow.end_run()
