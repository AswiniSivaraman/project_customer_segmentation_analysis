"""
Project_Clickstream_Conversion/
│
├── .zen/                              # ZenML configuration directory
│   ├── local_stores/
│   │   ├── c6b32bb5-3559-4a2c-b63d-9276842853e2/
│   │   ├── default_zen_store/
│   │   │   └── zenml.db
│   └── config.yaml
│
├── airflow/                            # Airflow DAGs for automation
│   └── dag_airflow.py
│
├── data/                               # Contains datasets for training and testing
│   ├── bulk_prediction_data.csv
│   ├── data description.txt
│   ├── test_data.csv
│   └── train_data.csv
│
├── docker/                             # Docker-related configurations
│   ├── mlruns/
│   │   ├── .trash/
│   │   └── models/
│   ├── docker.mainpipeline
│   ├── docker.mlflow
│   └── docker.streamlit
│
├── documents/                          # Contains documentation files
│   └── Errors.docx
│
├── images/                             # Contains images for Streamlit UI
│   ├── bg_image.png
│   └── logo.png
│
├── mlruns/                             # MLflow experiment tracking runs
│   ├── .trash/
│   ├── 0/
│   ├── 550180154531703585/
│   ├── 630753251537639831/
│   ├── 800721595944702217/
│   ├── 877744070416553456/
│   ├── 976576929185389185/
│   └── models/
│
├── models/                             # Saved ML models
│   ├── best_models/
│   │   └── best_models_file.csv
│   ├── classification_models/
│   │   ├── DecisionTreeClassifier.pkl
│   │   ├── GaussianNB.pkl
│   │   ├── GradientBoostingClassifier.pkl
│   │   ├── KNeighborsClassifier.pkl
│   │   ├── LogisticRegression.pkl
│   │   ├── MLPClassifier.pkl
│   │   ├── RandomForestClassifier.pkl
│   │   ├── SVC.pkl
│   │   └── XGBoostClassifier.pkl
│   ├── clustering_models/
│   │   ├── AgglomerativeClustering.pkl
│   │   ├── DBSCAN.pkl
│   │   ├── GaussianMixture.pkl
│   │   └── KMeans.pkl
│   └── regression_models/
│       ├── DecisionTreeRegressor.pkl
│       ├── ElasticNet.pkl
│       ├── GradientBoostingRegressor.pkl
│       ├── LassoRegression.pkl
│       ├── LinearRegression.pkl
│       ├── MLPRegressor.pkl
│       ├── PolynomialRegression.pkl
│       ├── RandomForestRegressor.pkl
│       ├── RidgeRegression.pkl
│       ├── SVR.pkl
│       └── XGBoostRegressor.pkl
│
├── notebook/                           # Contains Jupyter Notebooks
│   └── EDA.ipynb                        # Exploratory Data Analysis Notebook
│
├── pipelines/                          # ZenML pipeline definitions
│   ├── __pycache__/
│   ├── classification_pipeline.py
│   ├── clustering_pipeline.py
│   └── regression_pipeline.py
│
├── requirements doc/                   # Project requirement documents
│   ├── Clickstream-customer conversion.docx
│   └── Clickstream-customer conversion.pdf
│
├── src/                                # Core source code
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_feature_engineering.py
│   ├── data_feature_selection.py
│   ├── data_loading.py
│   ├── data_preprocessing.py
│   ├── find_best_model.py
│   ├── model_evaluation.py
│   └── model_training.py
│
├── steps/                              # ZenML steps for modular execution
│   ├── __pycache__/
│   ├── __init__.py
│   ├── best_model.py
│   ├── clean_data_steps.py
│   ├── clean_data.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── ingest_data.py
│   ├── rfe_feature_selection.py
│   ├── train_model.py
│   └── transforming_data.py
│
├── support/                            # Preprocessed data mappings and encoders
│   ├── add_ons/
│   │   ├── classification_page2_clothing_model_mapping.pkl
│   │   ├── classification_purchase_completed_mapping.pkl
│   │   ├── classification_season_mapping.pkl
│   │   ├── clustering_page2_clothing_model_mapping.pkl
│   │   ├── clustering_purchase_completed_mapping.pkl
│   │   ├── clustering_season_mapping.pkl
│   │   ├── regression_page2_clothing_model_mapping.pkl
│   │   ├── regression_purchase_completed_mapping.pkl
│   │   ├── regression_season_mapping.pkl
│   │   ├── classification_encoded_mappings.pkl
│   │   ├── classification_standard_scaler.pkl
│   │   ├── clustering_encoded_mappings.pkl
│   │   ├── clustering_standard_scaler.pkl
│   │   ├── regression_encoded_mappings.pkl
│   │   └── regression_standard_scaler.pkl
│
├── utils/                              # Utility and helper functions
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_transformation.py
│   ├── encoding_values.py
│   ├── get_result.py
│   ├── helper_func.py
│   └── rfe_selection.py
│
├── docker-compose.yml                   # Docker configuration
├── entrypoint.sh                         # Shell script for container entry
├── flow.py                               # Flow execution script
├── main_pipeline.py                      # Main pipeline execution script
├── main.py                               # Streamlit application main script
├── project_document.docx                 # Project documentation
├── requirements.txt                       # Python dependencies
└── test.ipynb                             # Jupyter notebook for testing
"""
