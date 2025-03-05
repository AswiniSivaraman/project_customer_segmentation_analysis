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
        # Initialize MLflow Experiment Tracking
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        with mlflow.start_run(run_name="Main_Pipeline_Run"):
            start_time = time.time()

            print(Client().active_stack.experiment_tracker.get_tracking_uri())

            # Run Regression Pipeline
            logging.info("Starting Regression Pipeline...")
            regression_pipeline(train_data, test_data)

            # # Run Classification Pipeline
            # logging.info("Starting Classification Pipeline...")
            # classification_pipeline(train_data, test_data)

            # # Run Clustering Pipeline
            # logging.info("Starting Clustering Pipeline...")
            # clustering_pipeline(train_data, test_data)

            # # Log Pipeline Execution Time
            # end_time = time.time()
            # execution_time = round(end_time - start_time, 2)
            # mlflow.log_metric("pipeline_execution_time", execution_time)
            # logging.info(f"Pipeline Execution Time: {execution_time} seconds")

    except Exception as e:
        logging.error("Error occurred when running the main pipeline")
        logging.exception("Full Exception Traceback:")
        raise e

if __name__ == "__main__":
    train_data_df = "data/train_data.csv"
    test_data_df = "data/test_data.csv"
    run_main_pipeline(train_data_df, test_data_df)




# import logging
# import mlflow
# import time
# from zenml.client import Client
# from pipelines.classification_pipeline import classification_pipeline
# from pipelines.regression_pipeline import regression_pipeline
# from pipelines.clustering_pipeline import clustering_pipeline
# from utils.get_result import fetch_best_model_outputs 

# def run_main_pipeline(train_data, test_data):
#     """
#     Runs classification, regression, and clustering pipelines sequentially with MLflow tracking.
#     """
#     try:
#         # Initialize MLflow Experiment Tracking
#         experiment_tracker = Client().active_stack.experiment_tracker
#         if experiment_tracker:
#             logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

#         with mlflow.start_run(run_name="Main_Pipeline_Run"):
#             start_time = time.time()

#             # # Run Regression Pipeline
#             # logging.info("Starting Regression Pipeline...")
#             # regression_pipeline(train_data, test_data)

#             # # Fetch Regression Model & Metrics
#             # best_regression_model, regression_metrics = fetch_best_model_outputs("regression_pipeline", "select_best_regression_model")
#             # if best_regression_model:
#             #     mlflow.log_param("best_regression_model", str(best_regression_model))
#             #     mlflow.log_metrics(regression_metrics)

#             #     print(f"Best Regression Model: {best_regression_model}")
#             #     print(f"Regression Metrics: {regression_metrics}")
#             # else:
#             #     print("No successful regression pipeline run found!")

#             # # Run Classification Pipeline
#             # logging.info("Starting Classification Pipeline...")
#             # classification_pipeline(train_data, test_data)

#             # # Fetch Classification Model & Metrics
#             # best_classification_model, classification_metrics = fetch_best_model_outputs("classification_pipeline", "select_best_classification_model")
#             # if best_classification_model:
#             #     mlflow.log_param("best_classification_model", str(best_classification_model))
#             #     mlflow.log_metrics(classification_metrics)

#             #     print(f"Best Classification Model: {best_classification_model}")
#             #     print(f"Classification Metrics: {classification_metrics}")
#             # else:
#             #     print("No successful classification pipeline run found!")

#             # Run Clustering Pipeline
#             logging.info("Starting Clustering Pipeline...")
#             clustering_pipeline(train_data, test_data)

#             # Fetch Clustering Model & Metrics
#             best_clustering_model, clustering_metrics = fetch_best_model_outputs("clustering_pipeline", "select_best_clustering_model")
#             if best_clustering_model:
#                 mlflow.log_param("best_clustering_model", str(best_clustering_model))
#                 mlflow.log_metrics(clustering_metrics)

#                 print(f"Best Clustering Model: {best_clustering_model}")
#                 print(f"Clustering Metrics: {clustering_metrics}")
#             else:
#                 print("No successful clustering pipeline run found!")

#             # # Log Pipeline Execution Time
#             # end_time = time.time()
#             # execution_time = round(end_time - start_time, 2)
#             # mlflow.log_metric("pipeline_execution_time", execution_time)

#             # logging.info(f"Pipeline Execution Time: {execution_time} seconds")

#     except Exception as e:
#         logging.error("Error occurred when running the main pipeline")
#         logging.exception("Full Exception Traceback:")
#         raise e

# if __name__ == "__main__":
#     train_data_df = "data/train_data.csv"
#     test_data_df = "data/test_data.csv"
#     run_main_pipeline(train_data_df, test_data_df)
