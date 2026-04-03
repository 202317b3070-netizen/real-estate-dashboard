import streamlit as st
 
def sidebar_controls(df):
    st.sidebar.title("⚙️ Control Panel")
 
    # Model selection
    st.sidebar.markdown("### 🧠 Model Settings")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest"]
    )
 
    # Currency
    st.sidebar.markdown("### 💱 Currency")
    currency = st.sidebar.selectbox(
        "Select Currency",
        ["USD", "INR"]
    )
 
    # Mode selection
    st.sidebar.markdown("### 🔄 Mode")
    mode = st.sidebar.radio(
        "Choose Mode",
        ["Predict Price", "Analyze Budget"]
    )
 
    # Inputs based on mode
    if mode == "Predict Price":
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
 
        occup = st.sidebar.slider(
            "Occupancy",
            float(df['AveOccup'].min()),
            float(df['AveOccup'].max()),
            3.0
        )
 
        return model_choice, currency, mode, income, rooms, occup, None
 
    else:
        st.sidebar.markdown("### 💰 Budget Input")
 
        budget = st.sidebar.number_input(
            "Enter Budget (₹)",
            min_value=100000,
            value=5000000
        )
 
        return model_choice, currency, mode, None, None, None, budget
 
