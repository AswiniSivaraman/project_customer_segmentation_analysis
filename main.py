import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.bulk_data_cleaning_streamlit import process_bulk_data  
import io
import base64
import time

# 1) Load the best model names
best_models_csv = os.path.join("models", "best_models", "best_models_file.csv")
best_models_df = pd.read_csv(best_models_csv)

class_model_name = best_models_df.loc[best_models_df['Model Type'] == 'classification', 'Best Model'].values[0]
reg_model_name = best_models_df.loc[best_models_df['Model Type'] == 'regression', 'Best Model'].values[0]
clust_model_name = best_models_df.loc[best_models_df['Model Type'] == 'clustering', 'Best Model'].values[0]
clust_model_name = "KMeans"

# 2) Load the .pkl models
classification_model_path = os.path.join("models", "classification_models", f"{class_model_name}.pkl")
regression_model_path = os.path.join("models", "regression_models", f"{reg_model_name}.pkl")
clustering_model_path = os.path.join("models", "clustering_models", f"{clust_model_name}.pkl")

with open(classification_model_path, "rb") as f:
    classification_model = pickle.load(f)
with open(regression_model_path, "rb") as f:
    regression_model = pickle.load(f)
with open(clustering_model_path, "rb") as f:
    clustering_model = pickle.load(f)

# 3) Load separate encoders
with open(os.path.join("support", "classification_encoded_mappings.pkl"), "rb") as f:
    classification_label_enc = pickle.load(f)
with open(os.path.join("support", "regression_encoded_mappings.pkl"), "rb") as f:
    regression_label_enc = pickle.load(f)
with open(os.path.join("support", "clustering_encoded_mappings.pkl"), "rb") as f:
    clustering_label_enc = pickle.load(f)

# 4) Load separate standard scalers
with open(os.path.join("support", "classification_standard_scaler.pkl"), "rb") as f:
    classification_scaler = pickle.load(f)
with open(os.path.join("support", "regression_standard_scaler.pkl"), "rb") as f:
    regression_scaler = pickle.load(f)
with open(os.path.join("support", "clustering_standard_scaler.pkl"), "rb") as f:
    clustering_scaler = pickle.load(f)

# Helper Functions

def label_to_code_strict(label_dict, chosen_label, field_name):
    """
    Convert the displayed label (chosen_label) back to its original code (key).
    For fields like 'page2_clothing_model', if the chosen label is one of the dictionary keys,
    then return its corresponding numeric code.
    Otherwise, iterate over items: if a dictionary value matches the chosen label, return its key.
    If no match is found, return the chosen label.
    """
    if chosen_label in label_dict:
        return label_dict[chosen_label]
    for k, v in label_dict.items():
        if v == chosen_label:
            return k
    return chosen_label

def scale_subset(scaler, df_subset):
    """
    For each column in df_subset, if the column was used during scaler training
    (i.e. exists in scaler.feature_names_in_), apply standard scaling using the stored mean and scale.
    Otherwise, leave the column unchanged.
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

# Manual mapping dictionaries for price_2 and page
price2_manual_map = {"yes": 1, "no": 2}
page_manual_map = {"Page 1": 1, "Page 2": 2, "Page 3": 3, "Page 4": 4, "Page 5": 5}

# Mappings for Non-Encoded Fields
weekend_map = {"Yes": 1, "No": 0}
season_map = {"Autumn": 0, "Summer": 1}

# 5) Streamlit setup & background setup
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

# Use os.path.join to get the background image path
bg_image_path = os.path.join("images", "bg_image.png")
set_background_in_main_area(bg_image_path)

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

# Use os.path.join for the sidebar logo
logo_path = os.path.join("images", "logo.png")
st.sidebar.image(logo_path, use_container_width =True)
st.sidebar.title("Customer Conversion Analysis")
st.sidebar.markdown("**Empowering E-commerce with Data Insights!**")

st.title("Customer Conversion Analysis for Online Shopping Using Clickstream Data")
st.markdown(
    """
    Welcome to the interactive web application for **Customer Conversion Analysis**.
    
    **How to Use:**
    1. **Manual Input**: Provide individual values for:
       - **Clustering**: features except month/day are shown (they're set internally).
       - **Regression**: all features except the target 'price'.
       - **Classification**: all features except the target 'purchase_completed'.
    2. **CSV Upload**: Upload your bulk data. The same pipeline is applied for clustering, regression, and classification.
    3. View results directly in this app.
    """
)

# Rename dictionary for CSV if needed
RENAME_DICT = {
    "page_1_main_category": "page1_main_category",
    "page_2_clothing_model": "page2_clothing_model"
}

# 6) Streamlit UI

input_mode = st.radio("Choose Input Method:", ["🖊️ Manual Input", "📂 Upload CSV Data"], index=0)

# ==================================================================================================================================================
#                                                            Manual Input Mode
# ==================================================================================================================================================
if input_mode == "🖊️ Manual Input":
    st.subheader("Enter Customer Details Manually")
    tab_clust, tab_reg, tab_class = st.tabs(["🔍 Customer Segmentation", "💰 Price Estimation", "🛍️ Purchase Prediction"])

    # Tab --> Clustering
    with tab_clust:
        st.markdown("### Customer Segmentation...")
        # Hard-code month=1, day=5 internally
        fixed_month_val = 1
        fixed_day_val = 5

        c1, c2 = st.columns(2)
        with c1:
            order_val = st.number_input("Sequence of Clicks (Order)", min_value=1, value=1, key="clust_order")
        with c2:
            country_dict_clust = clustering_label_enc.get("country", {})
            if country_dict_clust:
                country_sel = st.selectbox("Country", list(country_dict_clust.values()), key="clust_country")
            else:
                country_sel = st.text_input("Country", key="clust_country_text")

        c3, c4 = st.columns(2)
        with c3:
            session_id_val = st.text_input("Session ID", key="clust_session_id")
        with c4:
            page1_dict_clust = clustering_label_enc.get("page_1_main_category", {})
            if page1_dict_clust:
                page1_sel = st.selectbox("Main Product Category", list(page1_dict_clust.values()), key="clust_page1")
            else:
                page1_sel = st.text_input("Main Product Category", key="clust_page1_text")

        c5, c6 = st.columns(2)
        with c5:
            page2_dict_clust = clustering_label_enc.get("page2_clothing_model", {})
            if page2_dict_clust:
                page2_sel = st.selectbox("Clothing Model", list(page2_dict_clust.keys()), key="clust_page2")
            else:
                page2_sel = st.text_input("Clothing Model", key="clust_page2_text")
        with c6:
            colour_dict_clust = clustering_label_enc.get("colour", {})
            if colour_dict_clust:
                colour_sel = st.selectbox("Colour", list(colour_dict_clust.values()), key="clust_colour")
            else:
                colour_sel = st.text_input("Colour", key="clust_colour_text")

        c7, c8 = st.columns(2)
        with c7:
            location_dict_clust = clustering_label_enc.get("location", {})
            if location_dict_clust:
                location_sel = st.selectbox("Location", list(location_dict_clust.values()), key="clust_location")
            else:
                location_sel = st.text_input("Location", key="clust_location_text")
        with c8:
            model_photo_dict_clust = clustering_label_enc.get("model_photography", {})
            if model_photo_dict_clust:
                model_photo_sel = st.selectbox("Model Photography", list(model_photo_dict_clust.values()), key="clust_model_photo")
            else:
                model_photo_sel = st.text_input("Model Photography", key="clust_model_photo_text")

        c9, c10 = st.columns(2)
        with c9:
            price_val = st.number_input("Price", min_value=1.0, value=50.0, format="%.2f", key="clust_price")
        with c10:
            price2_dict_clust = clustering_label_enc.get("price_2", {})
            if price2_dict_clust:
                price2_sel = st.selectbox("Price Compared to Avg", list(price2_dict_clust.values()), key="clust_price2")
            else:
                price2_sel = st.text_input("Price Compared to Avg", key="clust_price2_text")

        c11, c12 = st.columns(2)
        with c11:
            page_dict_clust = clustering_label_enc.get("page", {})
            if page_dict_clust:
                page_sel = st.selectbox("Page", list(page_dict_clust.values()), key="clust_page")
            else:
                page_sel = st.text_input("Page", key="clust_page_text")
        with c12:
            purchase_completed_sel = st.selectbox("Purchase Completed (feature)", ["Yes", "No"], key="clust_purchased")

        c13, c14 = st.columns(2)
        with c13:
            is_weekend_sel = st.radio("Is Weekend?", ["Yes", "No"], key="clust_is_weekend")
        with c14:
            season_sel = st.selectbox("Season", list(season_map.keys()), key="clust_season")

        c15, c16 = st.columns(2)
        with c15:
            total_clicks_val = st.number_input("Total Clicks", min_value=0, value=0, key="clust_total_clicks")
        with c16:
            max_page_val = st.slider("Max Page Reached", 1, 5, 2, key="clust_max_page_reached")

        if st.button("Predict Customer Segment", key="clust_predict_btn"):
            month_val = fixed_month_val
            day_val = fixed_day_val

            try:
                session_id_final = int(session_id_val)
            except:
                session_id_final = 0

            if country_dict_clust:
                country_final = label_to_code_strict(country_dict_clust, country_sel, "Country")
            else:
                country_final = country_sel

            if page1_dict_clust:
                page1_final = label_to_code_strict(page1_dict_clust, page1_sel, "Main Product Category")
            else:
                page1_final = page1_sel

            if page2_dict_clust:
                page2_final = label_to_code_strict(page2_dict_clust, page2_sel, "Clothing Model")
            else:
                page2_final = page2_sel

            if colour_dict_clust:
                colour_final = label_to_code_strict(colour_dict_clust, colour_sel, "Colour")
            else:
                colour_final = colour_sel

            if location_dict_clust:
                location_final = label_to_code_strict(location_dict_clust, location_sel, "Location")
            else:
                location_final = location_sel

            if model_photo_dict_clust:
                model_photo_final = label_to_code_strict(model_photo_dict_clust, model_photo_sel, "Model Photography")
            else:
                model_photo_final = model_photo_sel

            price_final = price_val

            # Manually map price_2 using the manual mapping dictionary
            price2_final = price2_manual_map.get(str(price2_sel).lower(), price2_sel)
            # Manually map page using the manual mapping dictionary
            page_final = page_manual_map.get(page_sel, page_sel)

            purchase_completed_final = 1 if purchase_completed_sel == "Yes" else 0
            is_weekend_final = weekend_map[is_weekend_sel]
            season_final = season_map[season_sel]
            total_clicks_final = total_clicks_val
            max_page_final = max_page_val

            X_clust_manual = pd.DataFrame([{
                'month': month_val,
                'day': day_val,
                'order': order_val,
                'country': country_final,
                'session_id': session_id_final,
                'page1_main_category': page1_final,
                'page2_clothing_model': page2_final,
                'colour': colour_final,
                'location': location_final,
                'model_photography': model_photo_final,
                'price': price_final,
                'price_2': price2_final,
                'page': page_final,
                'purchase_completed': purchase_completed_final,
                'is_weekend': is_weekend_final,
                'season': season_final,
                'total_clicks': total_clicks_final,
                'max_page_reached': max_page_final
            }])

            X_clust_scaled = scale_subset(clustering_scaler, X_clust_manual)
            cluster_pred = clustering_model.predict(X_clust_scaled)
            st.session_state.cluster_pred = cluster_pred[0]
            st.session_state.session_id = session_id_val 
            st.success(f"Predicted Customer Segment: {st.session_state.cluster_pred}")

    # Tab --> Regression
    with tab_reg:
        st.markdown("### Price Estimation...")
        
        col1, col2 = st.columns(2)
        with col1:
            cluster_val = st.text_input(
                "Customer Segment (Cluster)",
                value=str(st.session_state.cluster_pred) if "cluster_pred" in st.session_state else "",
                disabled=True
            )
        with col2:
            session_id_reg = st.text_input(
                "Session ID",
                value=st.session_state.session_id if "session_id" in st.session_state else "",
                key="reg_session_id"
            )

        if "cluster_pred" not in st.session_state:
            st.info("Please run Clustering first.")
            predict_enabled_reg = False
        else:
            predict_enabled_reg = True

        col3, col4 = st.columns(2)
        with col3:
            country_dict_reg = regression_label_enc.get("country", {})
            if country_dict_reg:
                country_sel_reg = st.selectbox("Country", list(country_dict_reg.values()), key="reg_country")
            else:
                country_sel_reg = st.text_input("Country", key="reg_country_text")
        with col4:
            page1_dict_reg = regression_label_enc.get("page_1_main_category", {})
            if page1_dict_reg:
                page1_sel_reg = st.selectbox("Main Product Category", list(page1_dict_reg.values()), key="reg_page1")
            else:
                page1_sel_reg = st.text_input("Main Product Category", key="reg_page1_text")

        col5, col6 = st.columns(2)
        with col5:
            page2_dict_reg = regression_label_enc.get("page2_clothing_model", {})
            if page2_dict_reg:
                page2_sel_reg = st.selectbox("Clothing Model", list(page2_dict_reg.keys()), key="reg_page2")
            else:
                page2_sel_reg = st.text_input("Clothing Model", key="reg_page2_text")
        with col6:
            colour_dict_reg = regression_label_enc.get("colour", {})
            if colour_dict_reg:
                colour_sel_reg = st.selectbox("Colour", list(colour_dict_reg.values()), key="reg_colour")
            else:
                colour_sel_reg = st.text_input("Colour", key="reg_colour_text")

        col7, col8 = st.columns(2)
        with col7:
            model_photo_dict_reg = regression_label_enc.get("model_photography", {})
            if model_photo_dict_reg:
                model_photo_sel_reg = st.selectbox("Model Photography", list(model_photo_dict_reg.values()), key="reg_model_photo")
            else:
                model_photo_sel_reg = st.text_input("Model Photography", key="reg_model_photo_text")
        with col8:
            price2_dict_reg = regression_label_enc.get("price_2", {})
            if price2_dict_reg:
                price2_sel_reg = st.selectbox("Price Compared to Avg", list(price2_dict_reg.values()), key="reg_price2")
            else:
                price2_sel_reg = st.text_input("Price Compared to Avg", key="reg_price2_text")

        col9, col10 = st.columns(2)
        with col9:
            page_dict_reg = regression_label_enc.get("page", {})
            if page_dict_reg:
                page_sel_reg = st.selectbox("Page", list(page_dict_reg.values()), key="reg_page")
            else:
                page_sel_reg = st.text_input("Page", key="reg_page_text")
        with col10:
            is_weekend_reg = st.radio("Is Weekend?", ["Yes", "No"], key="reg_is_weekend")

        col11, col12 = st.columns(2)
        with col11:
            max_page_reached_reg = st.slider("Max Page Reached", 1, 5, 2, key="reg_max_page_reached")
        with col12:
            st.write("")

        if st.button("Estimate Price", disabled=not predict_enabled_reg, key="reg_predict_btn"):
            try:
                session_id_val_reg = int(session_id_reg)
            except:
                session_id_val_reg = 0

            if country_dict_reg:
                country_val_reg = label_to_code_strict(country_dict_reg, country_sel_reg, "Country")
            else:
                country_val_reg = country_sel_reg

            if page1_dict_reg:
                page1_val_reg = label_to_code_strict(page1_dict_reg, page1_sel_reg, "Main Product Category")
            else:
                page1_val_reg = page1_sel_reg

            if page2_dict_reg:
                page2_val_reg = label_to_code_strict(page2_dict_reg, page2_sel_reg, "Clothing Model")
            else:
                page2_val_reg = page2_sel_reg

            if colour_dict_reg:
                colour_val_reg = label_to_code_strict(colour_dict_reg, colour_sel_reg, "Colour")
            else:
                colour_val_reg = colour_sel_reg

            if model_photo_dict_reg:
                model_photo_val_reg = label_to_code_strict(model_photo_dict_reg, model_photo_sel_reg, "Model Photography")
            else:
                model_photo_val_reg = model_photo_sel_reg

            price2_val_reg = price2_manual_map.get(str(price2_sel_reg).lower(), price2_sel_reg)
            page_val_reg = page_manual_map.get(page_sel_reg, page_sel_reg)

            is_weekend_val_reg = weekend_map[is_weekend_reg]
            cluster_val_reg = st.session_state.cluster_pred

            X_reg_manual = pd.DataFrame([{
                'country': country_val_reg,
                'session_id': session_id_val_reg,
                'page1_main_category': page1_val_reg,
                'page2_clothing_model': page2_val_reg,
                'colour': colour_val_reg,
                'model_photography': model_photo_val_reg,
                'price_2': price2_val_reg,
                'page': page_val_reg,
                'is_weekend': is_weekend_val_reg,
                'max_page_reached': max_page_reached_reg,
                'cluster': cluster_val_reg
            }])

            X_reg_scaled = scale_subset(regression_scaler, X_reg_manual)
            y_pred_reg = regression_model.predict(X_reg_scaled)
            st.session_state.reg_price = y_pred_reg[0] 
            st.success(f"Estimated Price: ${y_pred_reg[0]:.2f}")

    # Tab --> Classification
    with tab_class:
        st.markdown("### Purchase Prediction...")

        cc_sess, cc_order = st.columns(2)
        with cc_sess:
            session_id_class = st.text_input("Session ID", value=st.session_state.session_id if "session_id" in st.session_state else "", key="class_session_id")
        with cc_order:
            order_class = st.number_input("Sequence of Clicks (Order)", min_value=1, value=1, key="class_order")

        cluster_val_class = st.text_input(
            "Customer Segment (Cluster)",
            value=str(st.session_state.cluster_pred) if "cluster_pred" in st.session_state else "",
            disabled=True,
            key="class_cluster"
        )

        if "cluster_pred" not in st.session_state:
            st.info("Please run Clustering first.")
            predict_enabled_class = False
        else:
            predict_enabled_class = True

        cc2, cc3 = st.columns(2)
        with cc2:
            page1_dict_class = classification_label_enc.get("page_1_main_category", {})
            if page1_dict_class:
                page1_sel_class = st.selectbox("Main Product Category", list(page1_dict_class.values()), key="class_page1")
            else:
                page1_sel_class = st.text_input("Main Product Category", key="class_page1_text")
            page2_dict_class = classification_label_enc.get("page2_clothing_model", {})
            if page2_dict_class:
                page2_sel_class = st.selectbox("Clothing Model", list(page2_dict_class.keys()), key="class_page2")
            else:
                page2_sel_class = st.text_input("Clothing Model", key="class_page2_text")
        with cc3:
            location_dict_class = classification_label_enc.get("location", {})
            if location_dict_class:
                location_sel_class = st.selectbox("Location", list(location_dict_class.values()), key="class_location")
            else:
                location_sel_class = st.text_input("Location", key="class_location_text")
            price_class = st.number_input("Price", min_value=1.0, value=st.session_state.reg_price if "reg_price" in st.session_state else 50.0, format="%.2f", key="class_price")

        cc4, cc5 = st.columns(2)
        with cc4:
            price2_dict_class = classification_label_enc.get("price_2", {})
            if price2_dict_class:
                price2_sel_class = st.selectbox("Price Compared to Avg", list(price2_dict_class.values()), key="class_price2")
            else:
                price2_sel_class = st.text_input("Price Compared to Avg", key="class_price2_text")
            page_dict_class = classification_label_enc.get("page", {})
            if page_dict_class:
                page_sel_class = st.selectbox("Page", list(page_dict_class.values()), key="class_page")
            else:
                page_sel_class = st.text_input("Page", key="class_page_text")
        with cc5:
            max_page_reached_class = st.slider("Max Page Reached", 1, 5, 2, key="class_max_page")

        if st.button("Predict Purchase Conversion", disabled=not predict_enabled_class, key="class_predict_btn"):
            if page1_dict_class:
                page1_val_class = label_to_code_strict(page1_dict_class, page1_sel_class, "Main Product Category")
            else:
                page1_val_class = page1_sel_class

            if page2_dict_class:
                page2_val_class = label_to_code_strict(page2_dict_class, page2_sel_class, "Clothing Model")
            else:
                page2_val_class = page2_sel_class

            if location_dict_class:
                location_val_class = label_to_code_strict(location_dict_class, location_sel_class, "Location")
            else:
                location_val_class = location_sel_class

            X_class_manual = pd.DataFrame([{
                'order': order_class,
                'page1_main_category': page1_val_class,
                'page2_clothing_model': page2_val_class,
                'location': location_val_class,
                'price': price_class,
                'price_2': price2_sel_class,  
                'page': page_sel_class,      
                'max_page_reached': max_page_reached_class,
                'cluster': st.session_state.cluster_pred
            }])

            X_class_manual["price_2"] = X_class_manual["price_2"].apply(lambda x: price2_manual_map.get(str(x).lower(), x) if isinstance(x, str) else x)
            X_class_manual["page"] = X_class_manual["page"].apply(lambda x: page_manual_map.get(x, x) if isinstance(x, str) else x)

            X_class_scaled = scale_subset(classification_scaler, X_class_manual)
            y_pred_class = classification_model.predict(X_class_scaled)

            if y_pred_class[0] == 1:
                st.success("Purchase will likely to complete")
            else:
                st.warning("Purchase will not likely to complete")

# ================================================================================================================================================================
#                                                                 CSV Upload Mode
# ================================================================================================================================================================
elif input_mode == "📂 Upload CSV Data":
    st.subheader("Upload Your Bulk Prediction CSV File")

    uploaded_file = st.file_uploader("📎 Choose a CSV file", type=["csv"], key="bulk_csv_uploader")
    
    if uploaded_file:
        file_bytes = uploaded_file.read()
        original_df = pd.read_csv(io.BytesIO(file_bytes))
        
        # Separate BytesIO objects for each pipeline
        file_clust = io.BytesIO(file_bytes)
        file_reg = io.BytesIO(file_bytes)
        file_class = io.BytesIO(file_bytes)
        
        with st.spinner("Processing CSV data..."):
            df_clust = process_bulk_data(file_clust, "clustering")
            df_reg = process_bulk_data(file_reg, "regression")
            df_class = process_bulk_data(file_class, "classification")
            time.sleep(2)
        
        if df_clust is None or df_reg is None or df_class is None:
            st.error("Data processing failed. Please check the logs.")
        else:
            # Extract the original session_id from the uploaded file.
            if "session_id" not in original_df.columns:
                st.error("No 'session_id' column found in the original CSV. Cannot proceed.")
                st.stop()
            original_session_ids = original_df["session_id"].reset_index(drop=True)
            
            tab_bulk_clust, tab_bulk_reg, tab_bulk_class = st.tabs(["🔍 Customer Segmentation", "💰 Price Estimation", "🛍️ Purchase Prediction"])
            
            # Bulk Customer Segmentation ( clustering )
            with tab_bulk_clust:
                st.markdown("### Bulk Customer Segmentation")
                
                if st.button("Run Bulk Clustering", key="bulk_clustering_btn"):
                    with st.spinner("Running clustering predictions..."):
                        cluster_labels = clustering_model.predict(df_clust)
                        df_clust["cluster"] = cluster_labels
                        
                        clust_result_df = df_clust.copy()
                        clust_result_df["session_id"] = original_session_ids
                        
                        st.success("Bulk Clustering Results:")
                        st.write(clust_result_df[["session_id", "cluster"]])
                        
                        cluster_counts = clust_result_df["cluster"].value_counts().sort_index()
                        total = cluster_counts.sum()

                        cluster_distribution = {c_label: (count / total) * 100.0 for c_label, count in cluster_counts.items()}
                        
                        # Store them in session_state for use in classification later
                        st.session_state["clust_result_df"] = clust_result_df
                        st.session_state["cluster_distribution"] = cluster_distribution
                        
                        # Plot cluster distribution with percentage annotations on each bar
                        fig_bar, ax_bar = plt.subplots()
                        bars = ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values)
                        ax_bar.set_xlabel("Customer Segment (Cluster)")
                        ax_bar.set_ylabel("Count")
                        dominant_percentage = (cluster_counts.max() / total) * 100
                        ax_bar.set_title(f"Cluster Distribution: {dominant_percentage:.1f}% of people will likely to buy the product")
                        
                        # Annotate each bar with its percentage
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            cluster_val = cluster_counts.index[i]
                            percentage = cluster_distribution[cluster_val]
                            ax_bar.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.1f}%', ha='center', va='bottom')
                        
                        st.pyplot(fig_bar)

            # Bulk Price Prediction ( Regression )
            with tab_bulk_reg:
                st.markdown("### Bulk Price Estimation")
                
                if st.button("Run Bulk Regression", key="bulk_regression_btn"):
                    with st.spinner("Running regression predictions..."):
                        X_reg_bulk = df_reg.drop(columns=["price"], errors="ignore")
                        preds_reg = regression_model.predict(X_reg_bulk)
                        df_reg["predicted_price"] = preds_reg
                        
                        reg_result_df = df_reg.copy()
                        reg_result_df["session_id"] = original_session_ids
                        
                        reg_agg = reg_result_df.groupby("session_id")["predicted_price"].sum().reset_index()
                        reg_agg["Estimated Price"] = reg_agg["predicted_price"].apply(lambda x: f"${x:.2f}")
                        reg_agg.drop(columns=["predicted_price"], inplace=True)
                        
                        st.success("Bulk Regression Results (Aggregated by Session ID):")
                        st.write(reg_agg)
            
            # Bulk Purchase Prediction ( classification )
            with tab_bulk_class:
                st.markdown("### Bulk Purchase Prediction")
                
                if st.button("Run Bulk Classification", key="bulk_classification_btn"):
                    with st.spinner("Running classification predictions..."):
                        X_class_bulk = df_class.drop(columns=["purchase_completed", "session_id"], errors="ignore")
                        preds = classification_model.predict(X_class_bulk)
                        
                        class_result_df = pd.DataFrame({"session_id": original_session_ids, "predicted_purchase": preds})
                        
                        # Merge in cluster membership from clustering step
                        clust_result_df = st.session_state["clust_result_df"]
                        cluster_distribution = st.session_state["cluster_distribution"]
                        
                        # Merge on session_id so each row gets its cluster
                        merged_df = class_result_df.merge(clust_result_df[["session_id", "cluster"]], on="session_id", how="left")
                        
                        # For each row, get cluster distribution from dictionary
                        merged_df["Cluster %"] = merged_df["cluster"].apply(lambda c: cluster_distribution.get(c, 0.0))
                        
                        # Define aggregator for repeated session_id rows
                        def aggregate_predictions(group):
                            total = len(group)
                            count_buy = (group["predicted_purchase"] == 1).sum()
                            prediction = 1 if count_buy >= (total / 2) else 0
                            percentage_buy = (count_buy / total) * 100
                            avg_cluster_pct = group["Cluster %"].mean()
                            return pd.Series({"Majority Prediction": prediction, "Purchase %": percentage_buy, "Avg Cluster %": avg_cluster_pct})
                        
                        agg_results = merged_df.groupby("session_id").apply(aggregate_predictions).reset_index()
                        
                        def format_prediction(row):
                            if row["Majority Prediction"] == 1:
                                return f"{row['Avg Cluster %']:.1f}% -> Purchase will likely to complete"
                            else:
                                return f"{row['Avg Cluster %']:.1f}% -> Purchase will not likely to complete"
                        
                        agg_results["Prediction"] = agg_results.apply(format_prediction, axis=1)
                        
                        st.success("Bulk Classification Results:")
                        st.write(agg_results[["session_id", "Prediction"]])
