import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
 
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Real Estate Analyzer", layout="wide")
 
st.title("🏠 Real Estate What-If Market Analyzer")
 
# -----------------------------
# LOAD REAL DATASET
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Price"] = data.target * 100000  # convert to realistic price
    return df
 
df = load_data()
 
# -----------------------------
# SHOW DATA
# -----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
 
# -----------------------------
# SELECT FEATURES
# -----------------------------
features = ['MedInc', 'AveRooms', 'AveOccup']
target = 'Price'
 
X = df[features]
y = df[target]
 
# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("🔧 What-If Analysis")
 
income = st.sidebar.slider("Median Income", float(X['MedInc'].min()), float(X['MedInc'].max()), 3.0)
rooms = st.sidebar.slider("Average Rooms", float(X['AveRooms'].min()), float(X['AveRooms'].max()), 5.0)
occup = st.sidebar.slider("Average Occupancy", float(X['AveOccup'].min()), float(X['AveOccup'].max()), 3.0)
 
# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[income, rooms, occup]])
prediction = model.predict(input_data)[0]
 
st.subheader("💰 Predicted House Price")
st.success(f"${int(prediction):,}")
 
# -----------------------------
# MODEL SCORE
# -----------------------------
score = model.score(X_test, y_test)
st.write(f"📈 Model R² Score: {round(score, 3)}")
 
# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.subheader("📊 Visual Insights")
 
col1, col2 = st.columns(2)
 
# Scatter Plot
with col1:
    st.write("Price vs Income")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['MedInc'], df['Price'])
    ax1.set_xlabel("Median Income")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)
 
# Heatmap
with col2:
    st.write("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
 
# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📌 Feature Importance")
 
coeff_df = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
st.bar_chart(coeff_df)
 
# -----------------------------
# SHOW FULL DATA
# -----------------------------
if st.checkbox("Show Full Dataset"):
    st.write(df)
 
# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built for Real Estate Analysis Project 🚀")
 
