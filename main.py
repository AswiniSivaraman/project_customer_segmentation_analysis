import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Clickstream Customer Conversion Analysis", layout="wide")

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
input_mode = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV Data"], index=None)

# If user selects Manual Input
if input_mode == "Manual Input":
    st.subheader("Enter Customer Browsing Details")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🛒 Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])
    
    with tab1:
        st.markdown("### Purchase Prediction (Classification)")
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.selectbox("Company", ["Amazon", "eBay", "Walmart", "Target"])
            country = st.selectbox("Country", ["USA", "UK", "India", "Germany", "France"])
            session_id = st.text_input("Session ID")
        
        with col2:
            browsing_duration = st.slider("Browsing Duration (mins)", 0, 120, 10)
            clicks = st.slider("Total Clicks", 1, 100, 5)
            bounce_rate = st.slider("Bounce Rate (%)", 0, 100, 50)
        
        if st.button("Predict Purchase Conversion"):
            st.success("Prediction Model Will Be Integrated Soon!")

    with tab2:
        st.markdown("### Revenue Estimation (Regression)")
        product_price = st.number_input("Product Price (USD)", min_value=1, value=100)
        num_items_viewed = st.slider("Number of Items Viewed", 1, 50, 5)
        time_spent = st.slider("Time Spent on Site (mins)", 1, 300, 15)
        
        if st.button("Estimate Revenue"):
            st.success("Regression Model Will Be Integrated Soon!")

    with tab3:
        st.markdown("### Customer Segmentation (Clustering)")
        engagement_score = st.slider("Engagement Score", 0, 100, 50)
        avg_order_value = st.slider("Average Order Value (USD)", 0, 1000, 250)
        
        if st.button("Segment Customer"):
            st.success("Clustering Model Will Be Integrated Soon!")

# If user selects CSV Upload
elif input_mode == "Upload CSV Data":
    st.subheader("Upload Your Clickstream Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())  # Display first few rows
        
        # Tabs for different analysis sections
        tab1, tab2, tab3 = st.tabs(["🛒 Purchase Prediction", "💰 Revenue Estimation", "📊 Customer Segmentation"])
    
        with tab1:
            st.markdown("### Purchase Prediction Results")
            st.write(df[["Session ID", "Total Clicks", "Bounce Rate"]].head())
        
        with tab2:
            st.markdown("### Revenue Estimation Results")
            st.write(df[["Session ID", "Revenue", "Browsing Duration"]].head())
        
        with tab3:
            st.markdown("### Customer Segmentation (Clustering Analysis)")
            
            # Histogram for browsing duration
            fig, ax = plt.subplots()
            sns.histplot(df["Browsing Duration"], bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of Browsing Duration")
            st.pyplot(fig)
            
            # Scatter plot for clicks vs. revenue
            fig, ax = plt.subplots()
            sns.scatterplot(x=df["Total Clicks"], y=df["Revenue"], hue=df["Cluster"], palette="viridis", ax=ax)
            ax.set_title("Customer Clusters Based on Clickstream Behavior")
            st.pyplot(fig)
