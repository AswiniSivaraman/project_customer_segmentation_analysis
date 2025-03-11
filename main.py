import logging
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page title and layout
st.set_page_config(page_title="Clickstream Customer Conversion Analysis", layout="wide")

# Function to Set Background Image from Local File
def set_background_contain(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(255, 255, 255, 0.8), 
                rgba(255, 255, 255, 0.8)
            ), 
            url("data:image/png;base64,{encoded_string}")
                no-repeat center center fixed;
            background-size: contain;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_contain("images/bg_image.png")


# -----------------------------------------------------------------------------
# 1) Load Encoded Mappings
# -----------------------------------------------------------------------------
def load_encoded_mappings():
    with open("support/encoded_mappings.pkl", "rb") as file:
        mappings = pickle.load(file)
    return mappings

encoded_mappings = load_encoded_mappings()

# Ensure correct key names for page1_main_category
if "page1_main_category" not in encoded_mappings:
    encoded_mappings["page1_main_category"] = encoded_mappings.get("page_1_main_category", {})

# -----------------------------------------------------------------------------
# 2) Build Decoded Mappings for Display
#    (Reverse most categories, keep page2_clothing_model/season as-is if needed)
# -----------------------------------------------------------------------------
decoded_mappings = {}
for key, mapping in encoded_mappings.items():
    if key in ['page2_clothing_model', 'season']:
        decoded_mappings[key] = mapping
    else:
        decoded_mappings[key] = {v: k for k, v in mapping.items()}

# -----------------------------------------------------------------------------
# 3) Sidebar
# -----------------------------------------------------------------------------
st.sidebar.image("images/logo.png", use_column_width=True)  # Replace with your logo file
st.sidebar.title("Customer Conversion Analysis")
st.sidebar.markdown("**Empowering E-commerce with Data Insights!**")

# -----------------------------------------------------------------------------
# 4) Main Title and Instructions
# -----------------------------------------------------------------------------
st.title("Customer Conversion Analysis for Online Shopping Using Clickstream Data")
st.markdown(
    """
    Welcome to the interactive web application for **Customer Conversion Analysis**.
    
    **How to Use:**
    1. Choose between **Manual Input** or **CSV Upload**.
    2. In **Manual Input**, provide the required values.
       - **Purchase Prediction (Classification):** Uses 9 features 
         ['month', 'order', 'session_id', 'page1_main_category', 'page2_clothing_model', 
          'location', 'price', 'page', 'max_page_reached'] (removed 'purchase_completed').
       - **Revenue Estimation (Regression):** Uses columns from your pkl (minus 'price') arranged in a 3×3 grid.
       - **Customer Segmentation (Clustering):** Displays visualizations only.
    3. In **CSV Upload**, upload your dataset to view the filtered columns and visualizations.
    """
)

# -----------------------------------------------------------------------------
# 5) Choose Input Mode
# -----------------------------------------------------------------------------
input_mode = st.radio("Choose Input Method:", ["🖊️ Manual Input", "📂 Upload CSV Data"], index=None)

# -----------------------------------------------------------------------------
# 6) Manual Input Mode
# -----------------------------------------------------------------------------
if input_mode == "🖊️ Manual Input":
    st.subheader("Enter Customer Browsing Details")
    tab1, tab2, tab3 = st.tabs(["🛍️ Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])
    
    # -------------------------------------------------------------------------
    # 6A) Purchase Prediction (Classification) Tab
    #     9 features: month, order, session_id, page1_main_category, 
    #                 page2_clothing_model, location, price, page, max_page_reached
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### 🛍️ Purchase Prediction (Classification)")
        
        # Create a 3×3 grid for the 9 fields
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        # --- ROW 1 ---
        with row1_col1:
            month_pred = st.selectbox(
                "📆 Month", 
                [
                    "January", "February", "March", "April", "May", 
                    "June", "July", "August", "September", "October", 
                    "November", "December"
                ],
                key="month_pred"
            )
        with row1_col2:
            order_pred = st.number_input("📌 Sequence of Clicks", min_value=1, value=1, key="order_pred")
        with row1_col3:
            session_id_pred = st.text_input("🆔 Session ID", key="session_id_pred")

        # --- ROW 2 ---
        with row2_col1:
            page1_main_category_pred = st.selectbox(
                "📁 Main Product Category", 
                list(decoded_mappings["page1_main_category"].keys()), 
                key="page1_main_category_pred"
            )
        with row2_col2:
            page2_clothing_model_pred = st.selectbox(
                "👕 Clothing Model", 
                list(decoded_mappings["page2_clothing_model"].keys()), 
                key="page2_clothing_model_pred"
            )
        with row2_col3:
            location_pred = st.selectbox(
                "📍 Photo Location", 
                list(decoded_mappings["location"].keys()), 
                key="location_pred"
            )

        # --- ROW 3 ---
        with row3_col1:
            price_pred = st.number_input("💲 Price", min_value=1, value=50, key="price_pred")
        with row3_col2:
            page_pred = st.selectbox(
                "📄 Page", 
                list(decoded_mappings["page"].keys()), 
                key="page_pred"
            )
        with row3_col3:
            max_page_reached_pred = st.slider("📄 Max Page Reached", 1, 5, 2, key="max_page_reached_pred")

        # Wide button at the bottom
        predict_btn = st.button("🚀 Predict Purchase Conversion", use_container_width=True)
        if predict_btn:
            st.success("Classification Model Will Be Integrated Soon!")
    
    # -------------------------------------------------------------------------
    # 6B) Revenue Estimation (Regression) Tab
    #     11 columns minus 'price', arranged in a 3×3 grid + final row
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 💰 Revenue Estimation (Regression)")
        
        # 3×3 grid for first 9 fields
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        # Row 1
        with row1_col1:
            country_rev = st.selectbox(
                "Country", 
                list(decoded_mappings.get("country", {}).keys()),
                key="country_rev"
            )
        with row1_col2:
            session_id_rev = st.text_input("Session ID", key="session_id_rev")
        with row1_col3:
            page1_main_category_rev = st.selectbox(
                "Main Product Category",
                list(decoded_mappings.get("page1_main_category", {}).keys()),
                key="page1_main_category_rev"
            )
        
        # Row 2
        with row2_col1:
            page2_clothing_model_rev = st.selectbox(
                "Clothing Model",
                list(decoded_mappings.get("page2_clothing_model", {}).keys()),
                key="page2_clothing_model_rev"
            )
        with row2_col2:
            colour_rev = st.selectbox(
                "Colour",
                list(decoded_mappings.get("colour", {}).keys()),
                key="colour_rev"
            )
        with row2_col3:
            location_rev = st.selectbox(
                "Location",
                list(decoded_mappings.get("location", {}).keys()),
                key="location_rev"
            )

        # Row 3
        with row3_col1:
            model_photography_rev = st.selectbox(
                "Model Photography",
                list(decoded_mappings.get("model_photography", {}).keys()),
                key="model_photography_rev"
            )
        with row3_col2:
            price_2_rev = st.selectbox(
                "Price Compared to Category Avg",
                list(decoded_mappings.get("price_2", {}).keys()),
                key="price_2_rev"
            )
        with row3_col3:
            page_rev = st.selectbox(
                "Page",
                list(decoded_mappings.get("page", {}).keys()),
                key="page_rev"
            )

        # Final row for the last 2 fields + wide button
        last_row_col1, last_row_col2 = st.columns([3,2])  # ratio for alignment
        with last_row_col1:
            is_weekend_rev = st.radio("Is Weekend?", ["Yes", "No"], key="is_weekend_rev")
            max_page_reached_rev = st.slider("Max Page Reached", 1, 5, 2, key="max_page_reached_rev")

        estimate_btn = st.button("Estimate Revenue", use_container_width=True)
        if estimate_btn:
            st.success("Regression Model Will Be Integrated Soon!")
    
    # -------------------------------------------------------------------------
    # 6C) Customer Segmentation (Clustering) Tab
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### 📊 Customer Segmentation (Clustering Analysis)")
        st.info("Clustering visualizations will be displayed here.")
        # Example Visualization: Distribution of Price (just a placeholder)
        fig, ax = plt.subplots()
        sample_values = [10, 20, 30, 40, 50]  # dummy data
        sns.histplot(sample_values, bins=5, kde=True, ax=ax)
        ax.set_title("Example Distribution (Clustering)")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# 7) CSV Upload Mode
# -----------------------------------------------------------------------------
elif input_mode == "📂 Upload CSV Data":
    st.subheader("Upload Your Clickstream Data")
    uploaded_file = st.file_uploader("📎 Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📜 Data Preview")
        st.write(df.head())
        
        # Tabs for CSV analysis
        tab1, tab2, tab3 = st.tabs(["🛍️ Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])
        
        # Classification columns (unchanged from your code)
        with tab1:
            st.markdown("### 🛍️ Purchase Prediction Results")
            selected_cols_class = [
                'month', 'order', 'session_id', 'page1_main_category',
                'page2_clothing_model', 'location', 'price', 'page',
                'max_page_reached', 'purchase_completed'
            ]
            common_cols_class = [col for col in selected_cols_class if col in df.columns]
            st.write(df[common_cols_class].head())
        
        # Regression columns
        with tab2:
            st.markdown("### 💰 Revenue Estimation Results")
            selected_cols_reg = [
                'country', 'session_id', 'page1_main_category', 'page2_clothing_model',
                'colour', 'location', 'model_photography', 'price_2',
                'page', 'is_weekend', 'max_page_reached'
            ]
            common_cols_reg = [col for col in selected_cols_reg if col in df.columns]
            st.write(df[common_cols_reg].head())
        
        # Clustering: Visualizations only
        with tab3:
            st.markdown("### 📊 Customer Segmentation (Clustering Analysis)")
            fig, ax = plt.subplots()
            if "price" in df.columns and "max_page_reached" in df.columns:
                sns.scatterplot(x=df["max_page_reached"], y=df["price"], ax=ax)
                ax.set_title("Price vs. Max Page Reached")
                st.pyplot(fig)
            else:
                st.info("Required columns for clustering visualization are missing.")
