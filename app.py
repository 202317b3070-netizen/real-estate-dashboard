import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import format_indian_price
from sidebar import sidebar_controls
from data import load_data
import plotly.express as px
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #1f2937;
}
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)
  
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Real Estate Analyzer", layout="wide")
 
st.title("🏠 Real Estate What-If Market Analyzer")
st.markdown("### Analyze housing prices with interactive predictions and insights")
 
# -----------------------------
# LOAD DATA
# -----------------------------
df=load_data()
model_choice, currency, mode, income, rooms, occup, budget = sidebar_controls(df)
st.subheader("📊 Dataset Overview")
 
st.write("### Shape of Dataset")
from sklearn.datasets import fetch_california_housing
 
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target * 100000
 
st.write(df.shape)
 
st.write("### Sample Data")
st.dataframe(df.head())

st.subheader("🧹 Data Cleaning")
 
missing_values = df.isnull().sum()
 
st.write("Missing Values in Dataset:")
st.write(missing_values)
 
if missing_values.sum() == 0:
    st.success("No missing values found ✅")
st.subheader("📈 Statistical Summary")
st.write(df.describe())

st.markdown("""
<div style="background: linear-gradient(90deg, #4facfe, #00f2fe);
            padding: 20px;
            border-radius: 12px;">
    <h1 style="color:white;">🏠 Real Estate What-If Market Analyzer</h1>
    <p style="color:white;">Analyze housing prices with interactive predictions</p>
</div>
""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
# 🏠 Real Estate What-If Market Analyzer
### 📊 Analyze housing prices with interactive predictions
""")

st.markdown("---")
city_coords = {
    "Delhi": (28.61, 77.23),
    "Mumbai": (19.07, 72.87),
    "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Pune": (18.52, 73.85),
    "Ahmedabad": (23.02, 72.57)
}
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
USD_TO_INR = 83
 
if currency == "INR":
    price = prediction * USD_TO_INR
    display_price = format_indian_price(price)
else:
    price = prediction
    display_price = f"$ {int(price):,}"
 
score = model.score(X_test, y_test)
if mode == "Predict Price":
    prediction = model.predict([[income, rooms, occup]])
elif mode == "Analyze Budget":
    st.write(f"Budget selected: {budget}")
 
# -----------------------------
# KPI METRICS
# -----------------------------
st.markdown("## 📊 Key Metrics")
 
col1, col2, col3 = st.columns(3)
 
col1.metric("💰 Predicted Price", display_price)
col2.metric("📈 Model Accuracy", f"{round(score, 2)}")
col3.metric("🏘 Avg Income", f"{round(df['MedInc'].mean(), 2)}")
 
# -----------------------------
# CHARTS
# -----------------------------
st.markdown("## 📉 Market Analysis")
 
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("### 💵 Income vs Price")
    fig = px.scatter(df, x="MedInc", y="Price", color="Price")
    st.plotly_chart(fig, use_container_width=True)
 
with col2:
    st.markdown("### 📊 Price Distribution")
    fig2 = px.histogram(df, x="Price")
    st.plotly_chart(fig2, use_container_width=True)
 
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
 
