import pandas as pd
import logging
from zenml import step
from src.data_feature_engineering import (
    target_feature,
    new_feature_1,
    new_feature_2,
    new_feature_3
)

@step
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML Step to apply feature engineering transformations.

    This step:
    - Creates a new target feature for classification (`purchase_completed`).
    - Creates time-based, season-based, and session-based features.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with new engineered features.
    """
    try:
        logging.info("Starting feature engineering...")

        # Step 1: Creating Target Feature for Classification Problem
        df = target_feature(df, col1="page", col2="price", col3="order", feature_name="purchase_completed")
        print("Target feature `purchase_completed` created.")
        logging.info("Target feature `purchase_completed` created.")

        # Step 2: Creating Time-Based Feature (Weekend vs. Weekday)
        df = new_feature_1(df, col1="day", feature_name="is_weekend")
        logging.info("Time-based feature `is_weekend` created.")
        print("Time-based feature `is_weekend` created.")

        # Step 3: Creating Season-Based Feature
        df = new_feature_2(df, col1="month", feature_name="season")
        logging.info("Season-based feature `season` created.")
        print("Season-based feature `season` created.")

        # Step 4: Creating Session-Based Features
        df = new_feature_3(df, col1="session_id", col2="order", transform_type="count", feature_name="total_clicks")
        logging.info("Session-based feature `total_clicks` created.")
        print("Session-based feature `total_clicks` created.")

        df = new_feature_3(df, col1="session_id", col2="page", transform_type="max", feature_name="max_page_reached")
        logging.info("Session-based feature `max_page_reached` created.")
        print("Session-based feature `max_page_reached` created.")

        logging.info("Feature engineering complete.")
        return df

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        logging.exception("Full Exception Traceback:")
        raise e



