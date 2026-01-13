import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Product Return Probability Predictor",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Product Return Probability Predictor")
st.write("Predict the probability that a product order will be returned.")

# -----------------------------
# Load saved model files
# -----------------------------
MODEL_PATH = "model"

with open(os.path.join(MODEL_PATH, "random_forest.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_PATH, "feature_order.pkl"), "rb") as f:
    feature_order = pickle.load(f)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Product Details")

price = st.number_input("Product Price", min_value=1.0, value=100.0)
rating = st.slider("Product Rating", min_value=1.0, max_value=5.0, step=0.1)

product_return_rate = st.slider(
    "Product Historical Return Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    help="Average return rate for this product"
)

avg_product_price = st.number_input(
    "Average Product Price",
    min_value=1.0,
    value=price,
    help="Average price of this product historically"
)

# -----------------------------
# Feature Engineering (same as notebook)
# -----------------------------
price_to_rating_ratio = price / (rating + 1)
high_price = 1 if price > avg_product_price else 0
low_rating = 1 if rating < 3 else 0

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_data = pd.DataFrame([[
    price,
    rating,
    price_to_rating_ratio,
    high_price,
    low_rating,
    product_return_rate,
    avg_product_price
]], columns=feature_order)

# Scale input
input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Return Risk"):
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.3:
        risk = "🟢 Low Risk"
    elif probability < 0.6:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader("Prediction Result")
    st.write(f"**Return Probability:** {probability:.2%}")
    st.write(f"**Risk Level:** {risk}")
