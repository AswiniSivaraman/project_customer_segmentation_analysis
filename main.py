import logging
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import time

# -------------------------
# 0) LOAD BEST MODEL NAMES
# -------------------------
best_models_csv = "models/best_models/best_models_file.csv"
best_models_df = pd.read_csv(best_models_csv)

class_model_name = best_models_df.loc[best_models_df['Model Type'] == 'classification', 'Best Model'].values[0]
reg_model_name = best_models_df.loc[best_models_df['Model Type'] == 'regression', 'Best Model'].values[0]
clust_model_name = best_models_df.loc[best_models_df['Model Type'] == 'clustering', 'Best Model'].values[0]

# -------------------------
# 1) LOAD THE .PKL MODELS
# -------------------------
classification_model_path = f"models/classification_models/{class_model_name}.pkl"
regression_model_path = f"models/regression_models/{reg_model_name}.pkl"
clustering_model_path = f"models/clustering_models/{clust_model_name}.pkl"

with open(classification_model_path, "rb") as f:
    classification_model = pickle.load(f)
with open(regression_model_path, "rb") as f:
    regression_model = pickle.load(f)
with open(clustering_model_path, "rb") as f:
    clustering_model = pickle.load(f)

# -------------------------
# 2) LOAD SEPARATE ENCODERS
# -------------------------
with open("support/classification_encoded_mappings.pkl", "rb") as f:
    classification_label_enc = pickle.load(f)
with open("support/regression_encoded_mappings.pkl", "rb") as f:
    regression_label_enc = pickle.load(f)

# -------------------------
# 2b) LOAD STANDARD SCALERS
# -------------------------
with open("support/classification_standard_scaler.pkl", "rb") as f:
    classification_scaler = pickle.load(f)
with open("support/regression_standard_scaler.pkl", "rb") as f:
    regression_scaler = pickle.load(f)
with open("support/clustering_standard_scaler.pkl", "rb") as f:
    clustering_scaler = pickle.load(f)

# -------------------------
# Helper Function for Partial Scaling
# -------------------------
def scale_subset(scaler, df_subset):
    """
    For each column in df_subset, if the column was used during scaler training
    (i.e. exists in scaler.feature_names_in_), apply standard scaling using the 
    stored mean and scale. Otherwise, leave the column unchanged.
    
    Returns a DataFrame with all columns (scaled or original) in the same order.
    """
    train_cols = scaler.feature_names_in_
    transformed = {}
    for col in df_subset.columns:
        if col in train_cols:
            index = list(train_cols).index(col)
            mean = scaler.mean_[index]
            scale = scaler.scale_[index]
            transformed[col] = (df_subset[col] - mean) / scale
        else:
            transformed[col] = df_subset[col]
    return pd.DataFrame(transformed, index=df_subset.index)

# -------------------------
# 3) STREAMLIT SETUP & BACKGROUND
# -------------------------
st.set_page_config(page_title="Clickstream Customer Conversion Analysis", layout="wide")

def set_background_in_main_area(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .block-container {{
            background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)),
                        url("data:image/png;base64,{encoded_string}") no-repeat center center;
            background-size: 100% 100%;
        }}
        [data-testid="stSidebar"] {{
            background-color: #efefed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_in_main_area("images/bg_image.png")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background-color: #dde4ec;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("images/logo.png", use_column_width=True)
st.sidebar.title("Customer Conversion Analysis")
st.sidebar.markdown("**Empowering E-commerce with Data Insights!**")

st.title("Customer Conversion Analysis for Online Shopping Using Clickstream Data")
st.markdown(
    """
    Welcome to the interactive web application for **Customer Conversion Analysis**.
    
    **How to Use:**
    1. **Manual Input**: Provide individual values for **Purchase Prediction** or **Price Estimation**.
       - **Purchase Prediction** determines if a customer is likely to make a purchase.
       - **Price Estimation** provides an estimated price value.
    2. **CSV Upload**: Upload your bulk data. A quick preview is shown, then you can run:
       - **Classification**: Shows whether each customer will purchase or not.
       - **Regression**: Estimates price in dollars.
       - **Clustering**: Groups similar customers and displays a scatter plot.
    3. View the results directly in this app, including tables and visualizations.
    """
)

# -------------------------
# 4) HELPER DICTS FOR NON-ENCODED FIELDS
# -------------------------
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
weekend_map = {"Yes": 1, "No": 0}

def label_to_code(label_dict, chosen_label, use_keys=False):
    """
    Convert a displayed label (value) back to its original code (key)
    or vice versa, depending on use_keys.
    """
    if use_keys:
        return label_dict.get(chosen_label, 0)
    for k, v in label_dict.items():
        if v == chosen_label:
            return k
    return 0

# -----------------------------------------
# FEATURE LISTS (Ensure spelling matches)
# -----------------------------------------
CLASSIFICATION_COLS = [
    'month','order','session_id','page1_main_category','page2_clothing_model',
    'location','price','page','max_page_reached'
]
# For regression, remove 'price' because it is the target to be predicted
REGRESSION_COLS = [
    'country','session_id','page1_main_category','page2_clothing_model',
    'colour','location','model_photography','price_2','page','is_weekend',
    'max_page_reached'
]
CLUSTERING_COLS = [
    'month','day','order','country','session_id','page1_main_category',
    'page2_clothing_model','colour','location','model_photography','price',
    'price_2','page','is_weekend','season','total_clicks','max_page_reached'
]

# If the CSV has slightly different column names, define a rename dictionary:
RENAME_DICT = {
    "page_1_main_category": "page1_main_category",
    "page_2_clothing_model": "page2_clothing_model"
}

# -------------------------
# 5) STREAMLIT UI
# -------------------------
input_mode = st.radio("Choose Input Method:", ["🖊️ Manual Input", "📂 Upload CSV Data"], index=0)

# =============================================================================
#                         MANUAL INPUT MODE
# =============================================================================
if input_mode == "🖊️ Manual Input":
    st.subheader("Enter Customer Browsing Details")
    tab1, tab2 = st.tabs(["🛍️ Purchase Prediction", "💰 Price Estimation"])

    # -------------------------------------------------------------------------
    # TAB 1: PURCHASE PREDICTION (CLASSIFICATION)
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🛍️ Purchase Prediction (Classification)")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7, col8, col9 = st.columns(3)

        with col1:
            month_pred = st.selectbox("📆 Month", list(month_map.keys()))
        with col2:
            order_pred = st.number_input("📌 Sequence of Clicks", min_value=1, value=1)
        with col3:
            session_id_pred = st.text_input("🆔 Session ID")

        with col4:
            page1_dict = classification_label_enc["page_1_main_category"]
            page1_labels = list(page1_dict.values())
            chosen_page1_label = st.selectbox("📁 Main Product Category", page1_labels)
        with col5:
            page2_dict = classification_label_enc["page2_clothing_model"]
            page2_labels = list(page2_dict.keys())
            chosen_page2_label = st.selectbox("👕 Clothing Model", page2_labels)
        with col6:
            loc_dict = classification_label_enc["location"]
            loc_labels = list(loc_dict.values())
            chosen_loc_label = st.selectbox("📍 Photo Location", loc_labels)
        with col7:
            price_pred = st.number_input("💲 Price", min_value=1, value=50)
        with col8:
            page_dict = classification_label_enc["page"]
            page_labels = list(page_dict.values())
            chosen_page_label = st.selectbox("📄 Page", page_labels)
        with col9:
            max_page_reached_pred = st.slider("📄 Max Page Reached", 1, 5, 2)

        if st.button("🚀 Predict Purchase Conversion", use_container_width=True):
            # Convert all labels/inputs to numeric codes
            month_val = month_map[month_pred]
            session_id_val = int(session_id_pred) if session_id_pred.isdigit() else 0
            page1_code = label_to_code(page1_dict, chosen_page1_label)
            page2_code = label_to_code(page2_dict, chosen_page2_label, use_keys=True)
            location_code = label_to_code(loc_dict, chosen_loc_label)
            page_code = label_to_code(page_dict, chosen_page_label)

            # Create a DataFrame for the single input row
            X_class = pd.DataFrame([{
                'month': month_val,
                'order': order_pred,
                'session_id': session_id_val,
                'page1_main_category': page1_code,
                'page2_clothing_model': page2_code,
                'location': location_code,
                'price': price_pred,
                'page': page_code,
                'max_page_reached': max_page_reached_pred
            }])

            # Use the helper to scale only the columns in X_class that are in the scaler
            X_class_scaled = scale_subset(classification_scaler, X_class)
            # Predict
            y_pred_class = classification_model.predict(X_class_scaled)
            if y_pred_class[0] == 1:
                st.success("This customer WILL make a purchase!")
            else:
                st.warning("This customer will NOT make a purchase.")

    # -------------------------------------------------------------------------
    # TAB 2: PRICE ESTIMATION (REGRESSION)
    #   We now PREDICT "price" as the target. So "price" is not an input.
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 💰 Price Estimation (Regression)")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7, col8, col9 = st.columns(3)

        with col1:
            country_dict = regression_label_enc["country"]
            country_labels = list(country_dict.values())
            chosen_country_label = st.selectbox("Country", country_labels)
        with col2:
            session_id_rev = st.text_input("Session ID")
        with col3:
            page1_dict_reg = regression_label_enc["page_1_main_category"]
            page1_labels_reg = list(page1_dict_reg.values())
            chosen_page1_label_reg = st.selectbox("Main Product Category", page1_labels_reg)
        with col4:
            page2_dict_reg = regression_label_enc["page2_clothing_model"]
            page2_labels_reg = list(page2_dict_reg.keys())
            chosen_page2_label_reg = st.selectbox("Clothing Model", page2_labels_reg)
        with col5:
            colour_dict_reg = regression_label_enc["colour"]
            colour_labels_reg = list(colour_dict_reg.values())
            chosen_colour_label_reg = st.selectbox("Colour", colour_labels_reg)
        with col6:
            location_dict_reg = regression_label_enc["location"]
            location_labels_reg = list(location_dict_reg.values())
            chosen_location_label_reg = st.selectbox("Location", location_labels_reg)
        with col7:
            model_photo_dict_reg = regression_label_enc["model_photography"]
            model_photo_labels_reg = list(model_photo_dict_reg.values())
            chosen_model_photo_label_reg = st.selectbox("Model Photography", model_photo_labels_reg)
        with col8:
            price2_dict_reg = regression_label_enc["price_2"]
            price2_labels_reg = list(price2_dict_reg.values())
            chosen_price2_label_reg = st.selectbox("Price Compared to Avg", price2_labels_reg)
        with col9:
            page_dict_reg = regression_label_enc["page"]
            page_labels_reg = list(page_dict_reg.values())
            chosen_page_label_reg = st.selectbox("Page", page_labels_reg)

        last_row_col1, last_row_col2 = st.columns([3,2])
        with last_row_col1:
            is_weekend_rev = st.radio("Is Weekend?", ["Yes", "No"])
            max_page_reached_rev = st.slider("Max Page Reached", 1, 5, 2)

        if st.button("Estimate Price", use_container_width=True):
            session_id_val = int(session_id_rev) if session_id_rev.isdigit() else 0
            country_code = label_to_code(country_dict, chosen_country_label)
            page1_code = label_to_code(page1_dict_reg, chosen_page1_label_reg)
            page2_code = label_to_code(page2_dict_reg, chosen_page2_label_reg, use_keys=True)
            colour_code = label_to_code(colour_dict_reg, chosen_colour_label_reg)
            location_code = label_to_code(location_dict_reg, chosen_location_label_reg)
            model_photo_code = label_to_code(model_photo_dict_reg, chosen_model_photo_label_reg)
            price_2_code = label_to_code(price2_dict_reg, chosen_price2_label_reg)
            page_code = label_to_code(page_dict_reg, chosen_page_label_reg)
            weekend_val = weekend_map[is_weekend_rev]

            # Create a DataFrame for the single input row
            X_reg = pd.DataFrame([{
                'country': country_code,
                'session_id': session_id_val,
                'page1_main_category': page1_code,
                'page2_clothing_model': page2_code,
                'colour': colour_code,
                'location': location_code,
                'model_photography': model_photo_code,
                'price_2': price_2_code,
                'page': page_code,
                'is_weekend': weekend_val,
                'max_page_reached': max_page_reached_rev
            }])

            # Scale only the columns in X_reg that are in the scaler
            X_reg_scaled = scale_subset(regression_scaler, X_reg)
            # Predict (this will be the predicted "price")
            y_pred_reg = regression_model.predict(X_reg_scaled)
            st.success(f"Estimated Price: ${y_pred_reg[0]:.2f}")

# =============================================================================
#                         CSV UPLOAD MODE
# =============================================================================
elif input_mode == "📂 Upload CSV Data":
    st.subheader("Upload Your Bulk Prediction CSV File")
    uploaded_file = st.file_uploader("📎 Choose a CSV file", type=["csv"])
    if uploaded_file:
        with st.spinner("Loading and processing CSV data..."):
            df = pd.read_csv(uploaded_file)
            time.sleep(2)  # Simulate processing delay

        # 1) Rename columns if needed
        rename_map = {}
        for old_col, new_col in RENAME_DICT.items():
            if old_col in df.columns:
                rename_map[old_col] = new_col
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        st.subheader("Data Preview (First Row)")
        st.write(df.head(1))

        # Define the columns for classification, regression, clustering
        CLASSIFICATION_COLS = [
            'month','order','session_id','page1_main_category','page2_clothing_model',
            'location','price','page','max_page_reached'
        ]
        # For regression, we exclude 'price' because it's our target
        REGRESSION_COLS = [
            'country','session_id','page1_main_category','page2_clothing_model',
            'colour','location','model_photography','price_2','page','is_weekend',
            'max_page_reached'
        ]
        CLUSTERING_COLS = [
            'month','day','order','country','session_id','page1_main_category',
            'page2_clothing_model','colour','location','model_photography',
            'price','price_2','page','is_weekend','season','total_clicks',
            'max_page_reached'
        ]

        tab1, tab2, tab3 = st.tabs(["🛍️ Classification", "💰 Regression", "📊 Clustering"])

        # ---------------------------------------------------------------------
        # BULK CLASSIFICATION
        # ---------------------------------------------------------------------
        with tab1:
            st.markdown("### Bulk Classification")
            if st.button("Run Bulk Classification"):
                with st.spinner("Running classification predictions..."):
                    if not set(CLASSIFICATION_COLS).issubset(df.columns):
                        missing_cols = set(CLASSIFICATION_COLS) - set(df.columns)
                        st.error(f"CSV is missing classification columns: {missing_cols}")
                    else:
                        X_class_bulk = df[CLASSIFICATION_COLS].copy()
                        # Scale using only the matching columns
                        X_class_bulk_scaled = scale_subset(classification_scaler, X_class_bulk)
                        preds = classification_model.predict(X_class_bulk_scaled)
                        result_class = pd.DataFrame({
                            "session_id": df["session_id"],
                            "Prediction": [
                                "WILL make a purchase" if p == 1 else "will NOT make a purchase"
                                for p in preds
                            ]
                        })
                        st.success("Bulk Classification Results:")
                        st.write(result_class)

        # ---------------------------------------------------------------------
        # BULK REGRESSION
        # ---------------------------------------------------------------------
        with tab2:
            st.markdown("### Bulk Regression (Predicting Price)")
            if st.button("Run Bulk Regression"):
                with st.spinner("Running regression predictions..."):
                    if not set(REGRESSION_COLS).issubset(df.columns):
                        missing_cols = set(REGRESSION_COLS) - set(df.columns)
                        st.error(f"CSV is missing regression columns: {missing_cols}")
                    else:
                        X_reg_bulk = df[REGRESSION_COLS].copy()
                        # Scale only the columns present
                        X_reg_bulk_scaled = scale_subset(regression_scaler, X_reg_bulk)
                        preds_reg = regression_model.predict(X_reg_bulk_scaled)
                        result_reg = pd.DataFrame({
                            "session_id": df["session_id"],
                            "Estimated Price": [f"${pred:.2f}" for pred in preds_reg]
                        })
                        st.success("Bulk Regression Results:")
                        st.write(result_reg)

        # ---------------------------------------------------------------------
        # BULK CLUSTERING
        # ---------------------------------------------------------------------
        with tab3:
            st.markdown("### Bulk Clustering")
            if st.button("Run Bulk Clustering"):
                with st.spinner("Running clustering predictions..."):
                    if not set(CLUSTERING_COLS).issubset(df.columns):
                        missing_cols = set(CLUSTERING_COLS) - set(df.columns)
                        st.error(f"CSV is missing clustering columns: {missing_cols}")
                    else:
                        X_clust_bulk = df[CLUSTERING_COLS].copy()
                        # Scale only the matching columns
                        X_clust_bulk_scaled = scale_subset(clustering_scaler, X_clust_bulk)
                        cluster_labels = clustering_model.predict(X_clust_bulk_scaled)
                        df["Cluster"] = cluster_labels
                        st.success("Bulk Clustering Results (first 10 rows):")
                        st.write(df[["session_id", "Cluster"]].head(10))

                        # Simple scatter plot example
                        fig, ax = plt.subplots()
                        sns.scatterplot(
                            x=df["max_page_reached"], 
                            y=df["total_clicks"], 
                            hue=df["Cluster"], 
                            ax=ax
                        )
                        ax.set_title("Clustering Visualization")
                        st.pyplot(fig)

                # ---- Add Pie Chart ----
                cluster_counts = df["Cluster"].value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%", startangle=90)
                ax2.set_title("Cluster Distribution")
                st.pyplot(fig2)
