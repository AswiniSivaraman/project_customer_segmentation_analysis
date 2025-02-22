import logging
from zenml import pipeline
from zenml.client import Client
from steps.ingest_data import ingest_train_data, ingest_test_data
from steps.clean_data import preprocessing_data
from steps.feature_engineering import apply_feature_engineering
from steps.feature_selection import select_feature_classification
from steps.transforming_data import scale_features, balance_train_data
from steps.train_model import train_classification
from steps.evaluate_model import evaluate_classification_step
from steps.best_model import select_best_classification_model

cat_cols = ['page2_clothing_model', 'season', 'purchase_completed']
cont_cols = ['month','day','order','country','session_id','page1_main_category','colour','location','model_photography','price','price_2','page','purchase_completed','is_weekend','total_clicks','max_page_reached']

@pipeline(enable_cache=False)
def classification_pipeline(train_data_path: str, test_data_path: str):
    """
    ZenML pipeline to train, evaluate, and select the best classification model.
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
        selected_features = select_feature_classification(df_train_featured,continuous_cols=cont_cols,categorical_cols=cat_cols,target_col="purchase_completed")

        # Step 5: Scale & Handle Imbalanced Data (Fit on Train, Transform on Train & Test)
        train_scaled, test_scaled = scale_features(df_train_featured, df_test_featured, numerical_cols=selected_features)
        X_train_balanced, y_train_balanced = balance_train_data(train_scaled, target_col="purchase_completed", method="smote")

        # Step 6: Train Model (Using Train Data Only)
        trained_classification_models = train_classification(X_train_balanced, y_train_balanced)

        # Step 7: Evaluate Models (Using Test Data Only)
        classification_eval_df = evaluate_classification_step(test_scaled, df_test_featured["purchase_completed"])

        # Step 8: Select Best Model
        best_classification_model, classification_metrics = select_best_classification_model(classification_eval_df)

        logging.info(f"Best Classification Model: {best_classification_model}")
        return best_classification_model, classification_metrics
    
    except Exception as e:
        logging.error(f'Error occured when running the classification pipeline')
        logging.exception('Full Exception Traceback:')
        raise e
