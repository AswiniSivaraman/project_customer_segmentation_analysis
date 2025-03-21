# Clickstream Customer Conversion Analysis

Customer Conversion Analysis for Online Shopping Using Clickstream Data is an end-to-end machine learning project designed to empower e-commerce businesses with actionable insights into customer behavior. Leveraging clickstream data, the project develops a Streamlit web application to predict customer purchase conversion, estimate potential revenue, and segment customers for personalized marketing.

## Table of Contents

- [Overview](#overview)
- [Business Use Cases](#business-use-cases)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Command-line Instructions](#command-line-instructions)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [Project Flow Diagram](#project-flow-diagram)

## Overview

In today’s competitive e-commerce landscape, understanding user behavior is key to optimizing marketing strategies and driving revenue growth. This project uses clickstream data to address three major tasks:

1. **Classification:** Predict whether a customer will complete a purchase based on their browsing behavior.
2. **Regression:** Estimate the potential revenue a customer might generate.
3. **Clustering:** Segment customers into distinct groups for targeted marketing and personalized product recommendations.

The interactive Streamlit web application provides a user-friendly interface for real-time and bulk predictions, enhancing decision-making for business stakeholders.

## Business Use Cases

- **Customer Conversion Prediction:** Identify potential buyers to improve conversion rates.
- **Revenue Forecasting:** Forecast customer spending to optimize pricing strategies.
- **Customer Segmentation:** Group users based on online behavior for personalized marketing.
- **Churn Reduction:** Detect users at risk of abandoning their carts and implement re-engagement strategies.
- **Product Recommendations:** Deliver personalized product suggestions based on browsing patterns.

## Features

- **Interactive Web Interface:** Streamlit application supporting both manual input and CSV upload modes for predictions.
- **Multiple ML Tasks:** Incorporates classification, regression, and clustering models to address various business problems.
- **Experiment Tracking & Reproducibility:** Utilizes ZenML pipelines and MLflow for robust experiment tracking and model management.
- **Comprehensive Data Processing:** Includes data ingestion, preprocessing, feature engineering, model training, and evaluation workflows.

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/clickstream-customer-conversion.git
   cd clickstream-customer-conversion
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Set up ZenML and MLflow:**
   
   Initialize ZenML and install the MLflow integration:
   ```bash
   zenml init
   zenml integration install mlflow -y
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml model-deployer register mlflow --flavor=mlflow
   zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
   ```
4. **Check ZenML Installation:**
   ```bash
   pip show zenml
   ```
5. **Log into ZenML locally:**
   ```bash
   zenml login --local --blocking
   ```

## Usage

1. **Run the Main Pipeline:**
   
   Execute the main pipeline script to run data ingestion, preprocessing, model training, and evaluation:
   ```bash
   python main_pipeline.py
   ```
2. **Launch the Streamlit Application:**
   
   Start the interactive web interface for real-time and bulk predictions:
   ```bash
   streamlit run main.py
   ```
3. **Start MLflow UI:**
   
   Open the MLflow UI by specifying the backend store URI (adjust the path as needed):
   ```bash
   mlflow ui --backend-store-uri "...\mlruns"
   ```

## Command-line Instructions

Below is a summary of the key commands used in this project:

```bash
zenml init
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
pip show zenml
zenml login --local --blocking
mlflow ui --backend-store-uri "...\mlruns"
python main_pipeline.py
streamlit run main.py
```

## Technical Details

### Data Processing
- Handles missing values, encodes categorical variables, and scales numerical features.
- Implements feature engineering including session-based and time-based indicators.

### Feature Selection
- Uses Recursive Feature Elimination (RFE), Chi-Square, ANOVA, and Pearson Correlation to identify important predictors for classification and regression tasks.

### Model Training & Evaluation
- Trains multiple models (e.g., Random Forest, XGBoost, Neural Networks) for each task.
- Evaluates models using:
  - **Classification:** Accuracy, Precision, Recall, F1 Score, ROC AUC
  - **Regression:** MAE, MSE, RMSE, R^2
  - **Clustering:** Silhouette Score, Davis-Bouldin Index

### Experiment Tracking
- Integrates MLflow with ZenML for logging model performance, hyperparameters, and artifacts, ensuring reproducibility.

## Dependencies

| Dependency         | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| pandas, numpy      | For data manipulation and numerical computations.                 |
| scikit-learn       | For implementing ML algorithms (classification, regression, clustering). |
| xgboost            | A high-performance gradient boosting framework.                    |
| zenml & mlflow     | For pipeline management and experiment tracking.                   |
| imbalanced-learn   | For handling imbalanced datasets.                                  |
| streamlit          | For building the interactive web application.                      |

Refer to **`requirements.txt`** for the complete list of dependencies.

### Project Flow Diagram

![image](https://github.com/user-attachments/assets/aad00c33-9a72-456f-8b45-63150ed20f72)


### **"To know more about this project refer `project_document.docx`."**

Deployed this application in **"Streamlit Cloud"** platform. Use this link to access the UI --> https://projectcustomersegmentationanalysis-p5muibk4gfezt4feneevmh.streamlit.app/


