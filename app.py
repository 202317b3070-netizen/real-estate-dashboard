import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utilsimport format_indian_price
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

@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Price"] = data.target * 100000
    return df
 
df = load_data()
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
# -----------------------------
# SIDEBAR (UPDATED UI)
# -----------------------------
st.sidebar.title("⚙️ Control Panel")
 
st.sidebar.markdown("### 🧠 Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest"]
)
 
st.sidebar.markdown("### 💱 Currency")
currency = st.sidebar.selectbox(
    "Select Currency",
    ["USD", "INR"]
)
 
st.sidebar.markdown("### 🏠 Property Features")
income = st.sidebar.slider(
    "Median Income",
    float(df['MedInc'].min()),
    float(df['MedInc'].max()),
    3.0
)
 
rooms = st.sidebar.slider(
    "Average Rooms",
    float(df['AveRooms'].min()),
    float(df['AveRooms'].max()),
    5.0
)
st.sidebar.markdown("### 📍 Location Filter")
 
lat = st.sidebar.slider(
    "Select Latitude",
    float(df['Latitude'].min()),
    float(df['Latitude'].max()),
    float(df['Latitude'].mean())
)
 
lon = st.sidebar.slider(
    "Select Longitude",
    float(df['Longitude'].min()),
    float(df['Longitude'].max()),
    float(df['Longitude'].mean())
)

filtered_df = df[
    (df['Latitude'].between(lat - 1, lat + 1)) &
    (df['Longitude'].between(lon - 1, lon + 1))
]
 
 
occup = st.sidebar.slider(
    "Occupancy",
    float(df['AveOccup'].min()),
    float(df['AveOccup'].max()),
    3.0
)

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
 
