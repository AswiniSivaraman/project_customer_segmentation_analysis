import logging
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Clickstream Customer Conversion Analysis", layout="wide")

# Load encoded mappings from pkl file
def load_encoded_mappings():
    with open("support/encoded_mappings.pkl", "rb") as file:
        mappings = pickle.load(file)
    return mappings

# Load the encoded mappings
encoded_mappings = load_encoded_mappings()

# Ensure correct key names for page1_main_category
if "page1_main_category" not in encoded_mappings:
    encoded_mappings["page1_main_category"] = encoded_mappings.get("page_1_main_category", {})

# Build decoded mappings:
# For most categories we reverse the mapping so that the original labels become the keys.
# For 'page2_clothing_model' and 'season', we use the original mapping as-is so that the user sees the proper labels.
decoded_mappings = {}
for key, mapping in encoded_mappings.items():
    if key in ['page2_clothing_model', 'season']:
        decoded_mappings[key] = mapping
    else:
        # Reverse mapping: keys become the original labels, values become their codes
        decoded_mappings[key] = {v: k for k, v in mapping.items()}

# Sidebar
st.sidebar.image("images/logo.png", use_column_width=True)  # Replace with your logo file
st.sidebar.title("Customer Conversion Analysis")
st.sidebar.markdown("**Empowering E-commerce with Data Insights!**")

# Main Title
st.title("Customer Conversion Analysis for Online Shopping Using Clickstream Data")
st.markdown(
    """
    Welcome to the interactive web application for **Customer Conversion Analysis**.
    
    **How to Use:**
    1. Choose between **Manual Input** or **CSV Upload**.
    2. If **Manual Input** is selected, enter required values to predict conversion.
    3. If **CSV Upload** is selected, upload a dataset for analysis and visualization.
    4. View insights from **Purchase Prediction, Revenue Estimation, and Customer Segmentation**.
    """
)

# Radio Button (Initially Unselected)
input_mode = st.radio("Choose Input Method:", ["🖊️ Manual Input", "📂 Upload CSV Data"], index=None)

# If user selects Manual Input
if input_mode == "🖊️ Manual Input":
    st.subheader("Enter Customer Browsing Details")

    # Tabs for different sections with icons
    tab1, tab2, tab3 = st.tabs(["🛍️ Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])

    with tab1:
        st.markdown("### 🛍️ Purchase Prediction (Classification)")
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("🌍 Country", list(decoded_mappings["country"].keys()))
            session_id = st.text_input("🆔 Session ID")
            order = st.number_input("📌 Sequence of Clicks", min_value=1, value=1)

        with col2:
            total_clicks = st.slider("🖱️ Total Clicks", 1, 200, 10)
            max_page_reached = st.slider("📄 Max Page Reached", 1, 5, 2)
            is_weekend = st.radio("📅 Is Weekend?", ["Yes", "No"])

        if st.button("🚀 Predict Purchase Conversion"):
            st.success("Prediction Model Will Be Integrated Soon!")

    with tab2:
        st.markdown("### 💰 Revenue Estimation (Regression)")
        price = st.number_input("💲 Product Price (USD)", min_value=1, value=50)
        price_2 = st.radio("📈 Price Compared to Category Average", ["Higher", "Lower"])
        num_items_viewed = st.slider("👀 Number of Items Viewed", 1, 50, 5)
        time_spent = st.slider("⏳ Time Spent on Site (mins)", 1, 300, 15)

        if st.button("📊 Estimate Revenue"):
            st.success("Regression Model Will Be Integrated Soon!")

    with tab3:
        st.markdown("### 📊 Customer Segmentation (Clustering)")
        col1, col2 = st.columns([1, 1])  # Equal column widths

        with col1:
            page1_main_category = st.selectbox("📁 Main Product Category", list(decoded_mappings["page1_main_category"].keys()))
            # For clothing model, use the original mapping so that models like 'A1' appear
            page2_clothing_model = st.selectbox("👕 Clothing Model", list(decoded_mappings["page2_clothing_model"].keys()))
            colour = st.selectbox("🎨 Product Colour", list(decoded_mappings["colour"].keys()))
            location = st.selectbox("📍 Photo Location on Page", list(decoded_mappings["location"].keys()))

        with col2:
            model_photography = st.radio("📸 Model Photography", list(decoded_mappings["model_photography"].keys()))
            # For season, use the original mapping to show 'Autumn'/'Summer'
            season = st.selectbox("🌤️ Season", list(decoded_mappings["season"].keys()))
            pages_visited = st.slider("📑 Pages Visited", 1, 5, 2)
            added_to_cart = st.slider("🛒 Added to Cart (%)", 0, 100, 20)

        if st.button("🔍 Segment Customer"):
            st.success("Clustering Model Will Be Integrated Soon!")

# If user selects CSV Upload
elif input_mode == "📂 Upload CSV Data":
    st.subheader("Upload Your Clickstream Data")
    uploaded_file = st.file_uploader("📎 Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📜 Data Preview")
        st.write(df.head())  # Display first few rows

        # Tabs for different analysis sections with icons
        tab1, tab2, tab3 = st.tabs(["🛍️ Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])

        with tab1:
            st.markdown("### 🛍️ Purchase Prediction Results")
            st.write(df[["session_id", "total_clicks", "is_weekend"]].head())

        with tab2:
            st.markdown("### 💰 Revenue Estimation Results")
            st.write(df[["session_id", "price", "price_2"]].head())

        with tab3:
            st.markdown("### 📊 Customer Segmentation (Clustering Analysis)")

            # Histogram for browsing duration
            fig, ax = plt.subplots()
            sns.histplot(df["total_clicks"], bins=20, kde=True, ax=ax)
            ax.set_title("🖱️ Distribution of Total Clicks")
            st.pyplot(fig)

            # Scatter plot for clicks vs. price
            fig, ax = plt.subplots()
            sns.scatterplot(x=df["total_clicks"], y=df["price"], hue=df.get("season", "Unknown"), palette="coolwarm", ax=ax)
            ax.set_title("🛍️ Customer Segmentation Based on Click Behavior & Price")
            st.pyplot(fig)
