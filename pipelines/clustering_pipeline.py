import logging
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

@pipeline(enable_cache=False)
def clustering_pipeline(train_data_path: str, test_data_path: str):
    """
    ZenML pipeline to train, evaluate, and select the best clustering model.
    """

    try:

        # Initialize MLflow Experiment Tracking
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        # Step 1: Ingest Data (Read train & test data from file paths)
        df_train = ingest_train_data(train_data_path)
        df_test = ingest_test_data(test_data_path)

        # Step 2: Clean Data
        df_train_cleaned = preprocessing_data(df_train)
        df_test_cleaned = preprocessing_data(df_test)

        # Step 3: Feature Engineering
        df_train_featured = apply_feature_engineering(df_train_cleaned)
        df_test_featured = apply_feature_engineering(df_test_cleaned)

        # Step 4: Feature Selection (Using Train Data Only)
        df_selected_train = select_feature_clustering(df_train_featured)
        df_selected_test = select_feature_clustering(df_test_featured)

        # Step 5: Scale Data (Fit on Train, Transform on Train & Test)
        train_scaled, test_scaled = scale_features(df_selected_train, df_selected_test, numerical_cols=df_selected_train.columns.tolist())

        # Step 6: Train Model (Using Train Data Only)
        trained_clustering_models = train_clustering(train_scaled)

        # Step 7: Evaluate Models (Using Test Data Only)
        clustering_eval_df = evaluate_clustering_step(test_scaled)

        # Step 8: Select Best Model
        best_clustering_model, clustering_metrics = select_best_clustering_model(clustering_eval_df)

        logging.info(f"Best Clustering Model: {best_clustering_model}")
        return best_clustering_model, clustering_metrics
    
    except Exception as e:
        logging.error(f'Error occured when running the clustering pipeline')
        logging.exception('Full Exception Traceback:')
        raise e