import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("🏠 Real Estate Price Predictor")

# Sample dataset (no upload needed)
data = {
    "area": [1000,1500,2000,2500,3000],
    "bedrooms": [2,3,3,4,5],
    "bathrooms": [1,2,2,3,4],
    "price": [200000,300000,400000,500000,650000]
}

df = pd.DataFrame(data)

X = df[['area','bedrooms','bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

st.sidebar.header("Enter Property Details")

area = st.sidebar.slider("Area",1000,3000,2000)
bedrooms = st.sidebar.slider("Bedrooms",1,5,3)
bathrooms = st.sidebar.slider("Bathrooms",1,4,2)

prediction = model.predict([[area,bedrooms,bathrooms]])

st.subheader("Predicted Price")
st.success(f"${int(prediction[0])}")

st.write("### Data Preview")
st.dataframe(df)
