import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression


def rfe_feature_selection(train_df: pd.DataFrame, target_column: str, n_features: int = 9, estimator=None):
    """
    Applies Recursive Feature Elimination (RFE) to select the top features from the training DataFrame.
    
    Args:
        train_df (pd.DataFrame): Training data containing features and target.
        target_column (str): Name of the target column.
        n_features (int): Number of features to select.
        estimator: A scikit-learn estimator to use for feature ranking. Defaults to LinearRegression() if None.
    
    Returns:
        list: Selected feature names.
    """
    if estimator == "regression":
        estimator = LinearRegression()
    else:
        estimator = LogisticRegression()
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    return X.columns[selector.support_].tolist()
