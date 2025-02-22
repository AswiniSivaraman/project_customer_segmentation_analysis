import logging
import pandas as pd
from zenml import pipeline
from zenml.client import Client
from steps.ingest_data import ingest_train_data, ingest_test_data
from steps.clean_data import preprocessing_data
from steps.feature_engineering import apply_feature_engineering
from steps.feature_selection import select_feature_regression
from steps.transforming_data import scale_features
from steps.train_model import train_regression
from steps.evaluate_model import evaluate_regression_step
from steps.best_model import select_best_regression_model

cat_cols = ['page2_clothing_model', 'season', 'purchase_completed']
cont_cols = ['month','day','order','country','session_id','page1_main_category','colour','location','model_photography','price','price_2','page','purchase_completed','is_weekend','total_clicks','max_page_reached']

@pipeline(enable_cache=False)
def regression_pipeline(train_data_path: str, test_data_path: str):
    """
    ZenML pipeline to train, evaluate, and select the best regression model.
    """
    try:
        # Initialize MLflow Experiment Tracking
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            logging.info(f"Using MLflow experiment tracker: {experiment_tracker.name}")

        # Step 1: Ingest Data (Separate Train & Test Files)
        df_train_artifact = ingest_train_data(train_data_path)
        df_test_artifact = ingest_test_data(test_data_path)
        print(f"DEBUG: df_train_artifact type: {type(df_train_artifact)}")
        print(f"DEBUG: df_train_artifact dir: {dir(df_train_artifact)}")
        print('data fetched')

        # Retrieve DataFrame from StepArtifact
        df_train = df_train_artifact.load()
        df_test = df_test_artifact.load()
        print('data dataframed')

        # Step 2: Clean Data
        df_train_cleaned = preprocessing_data(df_train).load() 
        df_test_cleaned = preprocessing_data(df_test).load()
        print('data cleaned')

        # Step 3: Feature Engineering
        df_train_featured = apply_feature_engineering(df_train_cleaned).load()
        df_test_featured = apply_feature_engineering(df_test_cleaned).load()
        print('data featured')

        # Step 4: Feature Selection (Using Train Data Only)
        selected_features = select_feature_regression(df_train_featured, continuous_cols=cont_cols, categorical_cols=cat_cols, target_col="price").load()
        print('data selected')

        # Step 5: Scale Data (Fit on Train, Transform on Train & Test)
        train_scaled, test_scaled = scale_features(df_train_featured, df_test_featured, numerical_cols=selected_features).load()
        print('data scaled')

        # Step 6: Train Model (Using Train Data Only)
        trained_regression_models = train_regression(train_scaled, df_train_featured["price"]).load()
        print('data trained')

        # Step 7: Evaluate Models (Using Test Data Only)
        regression_eval_df = evaluate_regression_step(test_scaled, df_test_featured["price"]).load()
        print('data evaluated')

        # Step 8: Select Best Model
        best_regression_model, regression_metrics = select_best_regression_model(regression_eval_df).load()
        print('data anything')

        logging.info(f"Best Regression Model: {best_regression_model}")
        return best_regression_model, regression_metrics
    
    except Exception as e:
        logging.error(f'Error occurred when running the regression pipeline')
        logging.exception('Full Exception Traceback:')
        raise e
