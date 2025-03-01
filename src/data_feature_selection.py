import pandas as pd
import logging
from scipy.stats import chi2_contingency,ttest_ind
from scipy.stats import f_oneway, pearsonr
from sklearn.feature_selection import VarianceThreshold

# to select the features, need to perform correlation analysis and hypothesis testing

def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform correlation analysis to select the features

    Args:
    df --> pandas dataframe

    Returns:
    pandas dataframe

    """
    try:
        logging.info('Performing correlation analysis to select the features')
        df_corr = df.corr()
        return df_corr
    except Exception as e:
        logging.error(f'Error occured when performing correlation analysis --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e


def feature_selection_classification(df:pd.DataFrame, continuous_cols:list, categorical_cols:list,  target_col:str) -> list:
    """
    Performs hypothesis testing (t-test and chi-squared test) to select the feature for classification problem

    Args:
    df --> Dataframe
    continuous_cols --> list contains continuous columns name
    categorical_cols --> list contains categorical columns name
    target --> Name of the dependent feature

    Retuns:
    list of selected columns 

    """

    try:

        logging.info("Starting Feature Selection for classification Problem.....")

        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform T-Test for continuous variables
        for col in continuous_cols:
            group1 = df[df[target_col] == df[target_col].unique()[0]][col]
            group2 = df[df[target_col] == df[target_col].unique()[1]][col]
            
            stat, p_value = ttest_ind(group1, group2, equal_var=False)  # T-Test
            results.append({"Feature": col, "Test": "T-Test", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Chi-Square Test for categorical variables
        for col in categorical_cols:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            results.append({"Feature": col, "Test": "Chi-Square", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})


        results_df = pd.DataFrame(results)

        # Filter the results to get only the features with significance < 0.05
        significant_features1 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()
        significant_features1.append(target_col)
        return significant_features1
    
    except Exception as e:
        logging.error('Error in Feature Selection for Classification Problem')
        logging.exception("Full Exception Traceback:")
        raise e


def feature_selection_regression(df:pd.DataFrame,  continuous_cols:list, categorical_cols:list, target_col:str) -> list :
    
    """
    Performs hypothesis testing (ANOVA F-Test and Pearson Correlation test) to select the feature for regression problem

    Args:
    df --> Dataframe
    continuous_cols --> list contains continuous columns name
    categorical_cols --> list contains categorical columns name
    target --> Name of the dependent feature

    Retuns:
    list of selected columns 
    
    """
    
    try:

        logging.info('Starting Feature Selection for Regression Problem.....')

        # Remove target column from feature lists
        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform ANOVA F-Test for categorical variables
        for col in categorical_cols:
            groups = [df[df[col] == cat][target_col] for cat in df[col].unique()]
            f_stat, p_value = f_oneway(*groups)
            results.append({"Feature": col, "Test": "ANOVA", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Pearson Correlation Test for continuous variables
        for col in continuous_cols:
            corr_coeff, p_value = pearsonr(df[col], df[target_col])
            results.append({"Feature": col, "Test": "Pearson Correlation", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Extract only significant features
        significant_features2 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()
        significant_features2.append(target_col)

        return significant_features2

    except Exception as e:
        logging.error('Error in Feature Selection for Regression Problem')
        logging.exception("Full Exception Traceback:")
        raise e
     

def feature_selection_clustering(df: pd.DataFrame, variance_threshold: float = 0.01) -> pd.DataFrame:
    """
    Perform feature selection for clustering models using Variance Threshold.

    Args:
        df (pd.DataFrame): The input dataset.
        variance_threshold (float): Minimum variance a feature must have to be kept.

    Returns:
        pd.DataFrame: Dataset with selected features for clustering.
    """
    try:
        logging.info("Starting feature selection for clustering using Variance Threshold.....")

        # Remove low-variance features
        var_selector = VarianceThreshold(threshold=variance_threshold)
        df_selected = pd.DataFrame(var_selector.fit_transform(df), columns=df.columns[var_selector.get_support()])
        logging.info(f"Removed low-variance features. Remaining features: {df_selected.shape[1]}")

        return df_selected

    except Exception as e:
        logging.error(f"Error in feature selection for clustering: {e}")
        logging.exception("Full Exception Traceback:")
        raise e
