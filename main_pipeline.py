import logging
import mlflow
import time
from zenml.client import Client
from pipelines.classification_pipeline import classification_pipeline
from pipelines.regression_pipeline import regression_pipeline
from pipelines.clustering_pipeline import clustering_pipeline
from utils.get_result import fetch_best_model_outputs 

def run_main_pipeline(train_data, test_data):
    """
    Runs classification, regression, and clustering pipelines sequentially with MLflow tracking.
    """
    try:
        start_time = time.time()

        # # Run Clustering Pipeline
        mlflow.set_experiment("Clustering Experiment")
        logging.info("Starting Clustering Pipeline...")
        clustering_pipeline(train_data, test_data)

        # Run Regression Pipeline
        mlflow.set_experiment("Regression Experiment")
        logging.info("Starting Regression Pipeline...")
        regression_pipeline(train_data, test_data)

        # Run Classification Pipeline
        mlflow.set_experiment("Classification Experiment")
        logging.info("Starting Classification Pipeline...")
        classification_pipeline(train_data, test_data)

        # Log Pipeline Execution Time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        logging.info(f"Pipeline Execution Time: {execution_time} seconds")

    except Exception as e:
        logging.error("Error occurred when running the main pipeline")
        logging.exception("Full Exception Traceback:")
        raise e
    
    finally:
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    train_data_df = "data/train_data.csv"
    test_data_df = "data/test_data.csv"
    run_main_pipeline(train_data_df, test_data_df)
