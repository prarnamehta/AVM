#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("best_scaler.pkl")

st.title("🏠 Automated Valuation Model (AVM)")

st.markdown("### Enter Property Features")

# Input fields
resam = st.selectbox("Property Use Code_RESAM - APARTMENTS MULTIPLE USE", ["Yes", "No"])
resap = st.selectbox("Property Use Code_RESAP - APARTMENTS", ["Yes", "No"])
water_frontage = st.number_input("Water Frontage Measurement", min_value=0.0, value=0.0)
land_area = st.number_input("Assessed Land Area", min_value=0.0, value=0.0)
zoning_freq = st.number_input("Zoning Frequency", min_value=0.0, value=0.0)

# Convert categorical inputs to binary
resam_val = 1 if resam == "Yes" else 0
resap_val = 1 if resap == "Yes" else 0

# Create input array
input_data = np.array([[resam_val, water_frontage, land_area, resap_val, zoning_freq]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Assessed Value"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Assessed Value: ${prediction:,.2f}")

