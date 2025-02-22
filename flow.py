"""
clickstream_project/
│
├── data/                              # Contains datasets
│   └── train.csv                      # Training dataset
│
├── src/                               # Main source folder for core functions
│   ├── __init__.py
│   ├── data_preprocessing.py          # Functions for data preprocessing
│   ├── data_split.py                  # Functions for splitting data into train/test sets
│   ├── model_training.py              # Functions for training machine learning models
│   └── model_evaluation.py            # Functions for evaluating models
│
├── steps/                             # ZenML steps for the pipeline
│   ├── __init__.py                    # Initialize steps as a Python package
│   ├── ingest_data.py                 # ZenML step for data ingestion
│   ├── clean_data.py                  # ZenML step for data cleaning and splitting
│   ├── train_model.py                 # ZenML step for model training
│   └── evaluate_model.py              # ZenML step for model evaluation
│
├── pipelines/                         # Folder for pipeline orchestration
│   └── clickstream_pipeline.py        # Main pipeline connecting all steps
│
├── utils/                             # Utility/helper functions
│   ├── __init__.py
│   └── helper_functions.py            # Helper functions for metrics, visualizations, etc.
│
├── streamlit_app/                     # Streamlit front-end application
│   ├── __init__.py
│   ├── app.py                         # Main Streamlit app for user interaction
│   └── components/                    # Components for Streamlit (sidebar, prediction UI)
│       ├── sidebar.py
│       └── prediction.py
│
├── mlflow_logs/                       # Folder for MLflow logs and artifacts
│
├── airflow_dags/                      # Placeholder for future Airflow DAGs
│   └── clickstream_dag.py             # Airflow DAG file for integration
│
├── Dockerfile                         # Dockerfile for containerization
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation

"""