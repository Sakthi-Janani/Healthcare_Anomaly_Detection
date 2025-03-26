import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load trained model and encoders
with open("isolation_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoded.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("mappings.pkl", "rb") as file:
    mappings = pickle.load(file)

df = pd.read_csv("processed_data.csv")  # Load processed data

# Reverse mapping dictionaries for display
reverse_mappings = {col: {v: k for k, v in mappings[col].items()} for col in mappings}

# ---- USER AUTHENTICATION ---- #
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "password123"
}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():
    """Login Page for User Authentication"""
    st.title("🔐 Login ")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome, {username}!")
            
        else:
            st.error("❌ Invalid username or password!")

if not st.session_state["logged_in"]:
    login()
else:
    # ---- MAIN APP ---- #
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

    # Streamlit App Title
    st.title("🚨 Anomalous Provider Detection")

    # Input Fields for Provider Information
    st.header("📌 Provider Information")

    # User Input with actual names
    provider_name = st.selectbox("🏥 Select Provider", df["Provider_Name"].unique())
    procedure_name = st.selectbox("💉 Select Procedure", df["Procedure"].unique())
    avg_covered_charges = st.number_input("💰 Average Covered Charges ($)", min_value=0.0, value=50000.0)
    average_total_payments = st.number_input("💳 Average Total Payments ($)", min_value=0.0, value=20000.0)
    total_discharge = st.number_input("📊 Total Discharges", min_value=0, value=100)

    # Convert selected names back to encoded values
    procedure_name = str(procedure_name).strip()
    provider_name = provider_name.strip()

    # Convert all values to strings before stripping
    procedure_mapping = {str(v).strip(): k for k, v in enumerate(df["Procedure"].unique())}
    reverse_mappings["Procedure"] = procedure_mapping

    encoded_procedure = procedure_mapping.get(procedure_name)

    if encoded_procedure is None:
        st.error(f"❌ Selected Procedure '{procedure_name}' not found in mappings!")
        st.write("🔍 Available Procedures:", list(procedure_mapping.keys()))
        st.stop()

    # ✅ Fix for Provider Name Mapping
    provider_mapping = {str(v).strip(): k for k, v in enumerate(df['Provider_Name'].unique())}
    reverse_mappings["Provider_Name"] = provider_mapping

    encoded_provider_name = provider_mapping.get(provider_name)

    if encoded_provider_name is None:
        st.error(f"❌ Selected Provider '{provider_name}' not found in mappings!")
        st.write("🔍 Available Providers:", list(provider_mapping.keys()))
        st.stop()

    # Prepare input for model
    final_input = np.array([[float(encoded_procedure), float(encoded_provider_name), 
                             float(total_discharge), float(avg_covered_charges), float(average_total_payments)]])

    # Predict Anomaly
    if st.button("🔍 Detect Anomaly"):
        prediction = model.predict(final_input)
        decision_score = model.decision_function(final_input)[0]  # Get decision function score
        is_anomalous = prediction[0] == -1        

        if is_anomalous:
            st.error(f"🚨 Alert! This provider is flagged as **ANOMALOUS**.\n\n📊 **Decision Score:** {round(decision_score, 4)}")
        else:
            st.success(f"✅ This provider appears **NORMAL**.\n\n📊 **Decision Score:** {round(decision_score, 4)}")

            


  