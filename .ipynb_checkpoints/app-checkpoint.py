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
# Load dataset (for insights)
# -----------------------------
DATA_PATH = "data/product_return_prediction.csv"
df = pd.read_csv(DATA_PATH)

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

# =====================================================
# SECTION 1: RETURN RISK PREDICTION
# =====================================================
st.markdown("---")
st.header("🔮 Predict Return Risk")

# -----------------------------
# User Inputs
# -----------------------------
price = st.number_input("Product Price", min_value=1.0, value=100.0)
rating = st.slider("Product Rating", min_value=1.0, max_value=5.0, step=0.1)

product_return_rate = st.slider(
    "Product Historical Return Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    help="Past return rate of this product"
)

avg_product_price = st.number_input(
    "Average Product Price",
    min_value=1.0,
    value=price,
    help="Historical average price of the product"
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
    st.metric("Return Probability", f"{probability:.2%}")
    st.write(f"**Risk Level:** {risk}")

# =====================================================
# SECTION 2: BUSINESS INSIGHTS DASHBOARD
# =====================================================
st.markdown("---")
st.header("📊 Business Insights Dashboard")

# -----------------------------
# Insight 1: Return Distribution
# -----------------------------
st.subheader("🔁 Return vs Non-Return Distribution")
return_counts = df['return_status'].value_counts().rename(
    {0: "Not Returned", 1: "Returned"}
)
st.bar_chart(return_counts)

# -----------------------------
# Insight 2: Return Rate by Price Segment
# -----------------------------
st.subheader("💰 Return Rate by Price Segment")

df['price_segment'] = pd.qcut(
    df['price'],
    q=4,
    labels=['Low', 'Medium-Low', 'Medium-High', 'High']
)

price_segment_returns = df.groupby('price_segment')['return_status'].mean()
st.bar_chart(price_segment_returns)

# -----------------------------
# Insight 3: Return Rate by Rating
# -----------------------------
st.subheader("⭐ Return Rate by Product Rating")
rating_returns = df.groupby('rating')['return_status'].mean()
st.line_chart(rating_returns)

# -----------------------------
# Insight 4: High Risk Products
# -----------------------------
st.subheader("🚨 High Risk Products")

high_risk_products = (
    df.groupby('product_id')['return_status']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(high_risk_products.rename("Return Rate"))

# -----------------------------
# Insight 5: Feature Importance
# -----------------------------
st.subheader("🧠 Feature Importance (Model Explainability)")

importance_df = pd.DataFrame({
    "Feature": feature_order,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
