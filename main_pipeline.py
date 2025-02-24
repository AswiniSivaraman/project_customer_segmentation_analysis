import logging
from pipelines.classification_pipeline import classification_pipeline
from pipelines.regression_pipeline import regression_pipeline
from pipelines.clustering_pipeline import clustering_pipeline
from zenml.client import Client
import mlflow
import time

# train_data = "data/train_data.csv"
# test_data = "data/test_data.csv"

def run_main_pipeline(train_data, test_data):
    """
    Runs classification, regression, and clustering pipelines sequentially with MLflow tracking.

    """
    try:

        # Initialize MLflow Experiment Tracking
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        with mlflow.start_run(run_name="Main_Pipeline_Run"):
            start_time = time.time()

            logging.info("Starting Regression Pipeline...")
            best_regression_model, regression_metrics = regression_pipeline(train_data, test_data)
            mlflow.log_param("best_regression_model", best_regression_model)
            print(regression_metrics)

            logging.info("Starting Classification Pipeline...")
            best_classification_model, classification_metrics = classification_pipeline(train_data, test_data)
            mlflow.log_param("best_classification_model", best_classification_model)
            print(classification_metrics)

            logging.info("Starting Clustering Pipeline...")
            best_clustering_model, clustering_metrics = clustering_pipeline(train_data, test_data)
            mlflow.log_param("best_clustering_model", best_clustering_model)
            print(clustering_metrics)

            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            # Log Pipeline Execution Time
            mlflow.log_metric("pipeline_execution_time", execution_time)

            logging.info(f"Best Regression Model: {best_regression_model}")
            logging.info(f"Best Regression Model's Metrics: {regression_metrics}")
            logging.info(f"Best Classification Model: {best_classification_model}")
            logging.info(f"Best Classification Model's Metrics: {classification_metrics}")
            logging.info(f"Best Clustering Model: {best_clustering_model}")
            logging.info(f"Best Clustering Model's Metrics: {clustering_metrics}")

    except Exception as e:
        logging.error(f'Error occured when running the main pipeline')
        logging.exception('Full Exception Traceback:')
        raise e

if __name__ == "__main__":
    train_data_df = "data/train_data.csv"
    test_data_df = "data/test_data.csv"
    run_main_pipeline(train_data_df, test_data_df)
