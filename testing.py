#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

# Load and preprocess the dataset
@st.cache_data
def load_and_prepare_data():
    ds = pd.read_csv("Assessment_Parcels_20251004.csv")

    # Drop empty columns
    empty_columns = ds.columns[ds.isnull().all()].tolist()
    ds = ds.drop(columns=empty_columns)

    # Clean numeric columns
    columns_to_clean = ['Street Number','Total Living Area','Total Assessed Value',
                        'Assessed Land Area','Water Frontage Measurement','Sewer Frontage Measurement',
                        'GISID','Roll Number','Dwelling Units']
    ds[columns_to_clean] = ds[columns_to_clean].replace('[\$,]', '', regex=True).astype(float)

    # Drop irrelevant rows
    ds['Roll Number'] = ds['Roll Number'].astype(object)
    roll_nos_to_drop = ['0107376627423 49.80160863413996', '9257376']
    ds = ds[~ds['Roll Number'].isin(roll_nos_to_drop)]

    # Handle missing values
    numerical_cols = ds.select_dtypes(include=['number']).columns
    categorical_cols = ds.select_dtypes(include=['object']).columns
    for col in numerical_cols:
        ds[col].fillna(ds[col].median(), inplace=True)
    for col in categorical_cols:
        mode_value = ds[col].mode()
        if not mode_value.empty:
            ds[col].fillna(mode_value[0], inplace=True)
        else:
            ds[col].fillna('Missing', inplace=True)

    # Filter residential properties
    categories_to_keep = ['RESSD - DETACHED SINGLE DWELLING', 'RESMB - RESIDENTIAL MULTIPLE BUILDINGS',
                          'RESSS - SIDE BY SIDE','RESMH - MOBILE HOME','RESRM - ROOMING HOUSE','RESDU - DUPLEX',
                          'RESTR - TRIPLEX','RESRH - ROW HOUSING','RESMC - MULTIFAMILY CONVERSION',
                          'RESGC - RESIDENTIAL GROUP CARE','RESOT - RESIDENTIAL OUTBUILDING',
                          'RESSU - RESIDENTIAL SECONDARY UNIT','RESMA - MULTIPLE ATTACHED UNITS',
                          'RESMU - RESIDENTIAL MULTIPLE USE','RESAM - APARTMENTS MULTIPLE USE',
                          'RESAP - APARTMENTS','CNRES - CONDO RESIDENTIAL']
    ds = ds[ds['Property Use Code'].isin(categories_to_keep)]

    # Drop irrelevant columns
    drop_cols = ['Street Number','Current Assessment Year','GISID','Centroid Lon','Centroid Lat',
                 'Full Address','Geometry','Detail URL','Assessed Value 1','Assessed Value 2',
                 'Assessed Value 3','Assessed Value 4','Assessed Value 5','Property Class 1',
                 'Property Class 2','Property Class 3','Property Class 4','Property Class 5',
                 'Status 1','Status 2','Status 3','Status 4','Status 5','Roll Number','Unit Number',
                 'Street Name','Street Suffix','Dwelling Units','Multiple Residences',
                 'Property Influences','Number Floors (Condo)']
    ds = ds.drop(columns=[col for col in drop_cols if col in ds.columns])

    # Encode categorical variables
    categorical_cols = ds.select_dtypes(include=['object', 'bool']).columns.tolist()
    category_counts = ds[categorical_cols].nunique()
    low_cardinality = category_counts[category_counts <= 20].index.tolist()
    high_cardinality = category_counts[category_counts > 20].index.tolist()

    df_encoded = pd.get_dummies(ds, columns=low_cardinality, drop_first=True)
    for col in high_cardinality:
        freq_encoding = ds[col].value_counts()
        df_encoded[col + '_freq'] = ds[col].map(freq_encoding)
    df_encoded.drop(columns=high_cardinality, inplace=True)

    return ds, df_encoded

# Train models and select the best one
@st.cache_data
def train_and_select_model(df_encoded):
    df_encoded = df_encoded.dropna(subset=["Total Assessed Value"])
    X = df_encoded.select_dtypes(include=[np.number]).drop(columns=["Total Assessed Value"])
    y = df_encoded["Total Assessed Value"]

    scalers = {
        "MinMax": MinMaxScaler(),
        "ZScore": StandardScaler()
    }

    models = {
        "RF": RandomForestRegressor(random_state=42),
        "XGB": XGBRegressor(random_state=42),
        "SVR": SVR(),
        "LR": LinearRegression()
    }

    best_model = None
    best_score = -np.inf
    best_scaler = None
    best_model_name = ""

    for scale_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = time.time()

            r2 = r2_score(y_test, y_pred)
            runtime = end_time - start_time

            # Select model with highest R² and lowest runtime
            score = r2 - 0.01 * runtime
            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_model_name = f"{model_name}-{scale_name}"

    return best_model, best_scaler, X.columns.tolist(), best_model_name

# Predict assessed value
def predict_value(model, scaler, feature_names, input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return prediction

# Streamlit UI
st.title("🏠 Property Valuation Predictor")

# Load data and train model
ds, df_encoded = load_and_prepare_data()
model, scaler, feature_names, model_name = train_and_select_model(df_encoded)

# User inputs
st.sidebar.header("Enter Property Details")
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
living_area = st.sidebar.number_input("Total Living Area (sq ft)", min_value=100, max_value=10000, value=1500)
rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=20, value=5)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
neighbourhoods = ds["Neighbourhood Area"].dropna().unique().tolist()
neighbourhood = st.sidebar.selectbox("Neighbourhood Area", sorted(neighbourhoods))

# Prepare input for prediction
neighbourhood_freq = ds["Neighbourhood Area"].value_counts().get(neighbourhood, 0)
input_data = []
for feature in feature_names:
    if feature == "Year Built":
        input_data.append(year_built)
    elif feature == "Total Living Area":
        input_data.append(living_area)
    elif feature == "Rooms":
        input_data.append(rooms)
    elif feature == "Bathrooms":
        input_data.append(bathrooms)
    elif feature == "Neighbourhood Area_freq":
        input_data.append(neighbourhood_freq)
    else:
        input_data.append(0)

# Predict and display result
if st.button("Predict Assessed Value"):
    predicted_value = predict_value(model, scaler, feature_names, input_data)
    st.success(f"💰 Predicted Assessed Value: ${predicted_value:,.2f}")
    st.caption(f"Model used: {model_name}")

