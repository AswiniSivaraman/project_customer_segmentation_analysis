import pandas as pd
import logging


def target_feature(df: pd.DataFrame, col1: str, col2: str, col3: str, feature_name: str) -> pd.DataFrame:
    """
    Create a new feature by combining three existing features

    Args:
    df --> pandas dataframe
    col1 --> column name 1
    col2 --> column name 2
    col3 --> column name 3
    feature_name --> name of the new feature

    Returns:
    pandas dataframe
    """
    try:
        logging.info(f'Creating a new feature called {feature_name} by combining existing features')
        df[feature_name] = df.apply(lambda row: 1 if (row[col1] == 5 and row[col2] > 0 and row[col3] > 10) else 0, axis=1)
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e
    
# note col1 'page' , col2 'price', col3 'order'

# PAGE indicates the stage of the user's journey through the e-commerce website.
# Typically, Page 5 could represent a checkout or purchase confirmation page, which is the last step in the process before completing the purchase.
# If the user reaches Page 5, it strongly suggests that they are finalizing the purchase.


#  The PRICE column indicates the price of the product being viewed or purchased.
# If PRICE > 0, it confirms that the user interacted with a product that has a valid price.
# If PRICE == 0, it might be a non-purchasable item or a placeholder product view.
# If PRICE > 0, it suggests a real product interaction and increases the likelihood of a purchase event.


# 3. row['ORDER'] > 10
# Reasoning: ORDER represents the sequence of clicks during a session.
# The higher the ORDER value, the longer and more engaged the session.
# ORDER <= 10 could mean the user browsed briefly and left without purchasing.
# ORDER > 10 suggests a deeper level of engagement, indicating the user is more likely to complete a purchase.
# Why > 10?
# Thresholds like 10 are chosen based on domain knowledge or empirical analysis.
# For example, in clickstream data, a session with more than 10 clicks often signifies serious interest and intent to purchase, especially if it ends at PAGE 5.


def new_feature_1(df: pd.DataFrame, col1: str, feature_name: str) -> pd.DataFrame:
    """
    Create a new feature to analyse the time-base feature based on an existing feature.

    Args:
    df --> pandas dataframe
    col1 --> column name

    Returns:
    pandas dataframe

    df['is_weekend'] = df['day'].apply(lambda x: 1 if x in [6, 7] else 0)

    """
    try:
        logging.info(f'Creating a new feature called {feature_name} based on an existing feature')
        df[feature_name] = df[col1].apply(lambda x: 1 if x in [6,7] else 0)
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e


def new_feature_2(df: pd.DataFrame, col1: str, feature_name: str) -> pd.DataFrame:
    """
    Create a new feature to analyse the seaspn-base feature based on an existing feature.

    Args:
    df --> pandas dataframe
    col1 --> column name

    Returns:
    pandas dataframe

    df['season'] = df['month'].apply(lambda x: 'Summer' if x in [4, 5, 6] else 'Autumn')

    """
    try:
        logging.info(f'Creating a new feature called {feature_name} based on an existing feature')
        df[feature_name] = df[col1].apply(lambda x: 'Summer' if x in [4,5,6] else 'Autumn')
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e


def new_feature_3(df: pd.DataFrame, col1: str, col2: str, transform_type: str,feature_name: str) -> pd.DataFrame:
    """
    Create a new feature to analyse the Session-base feature based on an existing feature.

    Args:
    df --> pandas dataframe
    col1 --> column name
    col2 --> column name
    transform_type --> transformation type

    Returns:
    pandas dataframe

    df['total_clicks'] = df.groupby('session_id')['order'].transform('count')
    df['max_page_reached'] = df.groupby('session_id')['page'].transform('max')

    """
    try:
        logging.info(f'Creating a new feature called {feature_name} based on an existing feature')
        df[feature_name] = df.groupby(col1)[col2].transform(transform_type)
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        logging.exception('Full Exception Traceback:')
        raise e



