import streamlit as st
import pandas as pd
import pickle

from clean_nlp import clean_data
from feature_engineering import add_features  


# -----------------------------
# LOAD ALL MODELS AT START
# -----------------------------
iraq_model      = pickle.load(open("iraq/iraq_xgb_model.pkl", "rb"))
iraq_features   = pickle.load(open("iraq/iraq_feature_list.pkl", "rb"))

leb_model       = pickle.load(open("lebanon/lebanon_xgb_model.pkl", "rb"))
leb_features    = pickle.load(open("lebanon/lebanon_feature_list.pkl", "rb"))

usa_model       = pickle.load(open("usa/usa_xgb_model.pkl", "rb"))
usa_features    = pickle.load(open("usa/usa_feature_list.pkl", "rb"))

# --------------------------------
# STREAMLIT UI
# --------------------------------

st.title("Real Estate Price Prediction")

country = st.selectbox("Select Country:", ["IRAQ", "LEBANON", "USA"])

area = st.number_input("Area (sqft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
city = st.text_input("Location")
property_type = st.text_input("Property Type")
description = st.text_area("Description")
amenities = st.text_area("Amenities (Python list format)")

if st.button("Predict"):

    # Build DataFrame from user input
    user_df = pd.DataFrame([{
        "country": country,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": city,
        "property_type": property_type,
        "description": description,
        "amenities": amenities
    }])

    # 1) CLEANING
    clean_row = clean_data(user_df)

    # 2) FEATURE ENGINEERING
    engineered_row = add_features(clean_row)

    # 3) SELECT MODEL AND FEATURE LIST
    if country == "IRAQ":
        model = iraq_model
        features = iraq_features
    elif country == "LEBANON":
        model = leb_model
        features = leb_features
    else:
        model = usa_model
        features = usa_features

    # 4) ALIGN COLUMNS
    X = engineered_row.reindex(columns=features, fill_value=0)

    # 5) PREDICT
    price = model.predict(X)[0]
    st.success(f"Predicted Price: ${price:,.0f}")