import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
 
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Real Estate Analyzer", layout="wide")
 
st.title("🏠 Real Estate What-If Market Analyzer")
st.markdown("### Analyze housing prices with interactive predictions and insights")
 
# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Price"] = data.target * 100000
    return df
 
df = load_data()
 
# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Controls")
 
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest"]
)
 
# ✅ ADD THIS HERE
currency = st.sidebar.selectbox(
    "Select Currency",
    ["USD", "INR"]
)
def format_indian_price(price):
    if price >= 10000000:
        return f"₹ {price/10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹ {price/100000:.2f} Lakh"
    else:
        return f"₹ {price:,.0f}"
 
income = st.sidebar.slider("Median Income", float(df['MedInc'].min()), float(df['MedInc'].max()), 3.0)
rooms = st.sidebar.slider("Average Rooms", float(df['AveRooms'].min()), float(df['AveRooms'].max()), 5.0)
occup = st.sidebar.slider("Average Occupancy", float(df['AveOccup'].min()), float(df['AveOccup'].max()), 3.0)
 
# -----------------------------
# MODEL
# -----------------------------
features = ['MedInc', 'AveRooms', 'AveOccup']
X = df[features]
y = df['Price']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()
 
model.fit(X_train, y_train)
 
prediction = model.predict([[income, rooms, occup]])[0]
score = model.score(X_test, y_test)
 
# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)
 
USD_TO_INR = 83
 
if currency == "INR":
    price = prediction * USD_TO_INR
    display_price = format_indian_price(price)
else:
    price = prediction
    display_price = f"$ {int(price):,}"
 
col1.metric("💰 Predicted Price", display_price)
col2.metric("📈 Model Accuracy (R²)", round(score, 3))
col3.metric("🏘 Avg Area Income", round(df['MedInc'].mean(), 2))
 
# -----------------------------
# CHARTS
# -----------------------------
st.subheader("📊 Market Insights")
 
col1, col2 = st.columns(2)
 
with col1:
    fig = px.scatter(df, x="MedInc", y="Price", title="Income vs Price")
    st.plotly_chart(fig)
 
with col2:
    fig2 = px.histogram(df, x="Price", title="Price Distribution")
    st.plotly_chart(fig2)
 
# -----------------------------
# WHAT-IF ANALYSIS TEXT
# -----------------------------
st.subheader("🔍 What-If Analysis Insight")
 
if income > df['MedInc'].mean():
    st.info("Higher income areas tend to have higher house prices.")
else:
    st.info("Lower income areas generally show lower house prices.")
 
# -----------------------------
# FEATURE IMPORTANCE (RF ONLY)
# -----------------------------
if model_choice == "Random Forest":
    st.subheader("📌 Feature Importance")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })
    st.bar_chart(importance.set_index("Feature"))
 
# -----------------------------
# DATA VIEW
# -----------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df)
 
# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Developed for Real Estate Analytics Project")
 
