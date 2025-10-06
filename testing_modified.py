#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Automated Valuation Model", layout="wide")
st.title("🏠 Automated Valuation Model (AVM)")

uploaded_file = st.file_uploader("Upload raw CSV file", type=["csv"])
if uploaded_file:
    ds = pd.read_csv(uploaded_file)

    # Drop empty columns
    empty_columns = ds.columns[ds.isnull().all()].tolist()
    ds.drop(columns=empty_columns, inplace=True)

    # Clean numeric columns with special characters
    columns_to_clean = ['Street Number','Total Living Area','Total Assessed Value',
                        'Assessed Land Area','Water Frontage Measurement','Sewer Frontage Measurement',
                        'GISID','Roll Number','Dwelling Units']
    for col in columns_to_clean:
        if col in ds.columns:
            ds[col] = ds[col].replace('[\$,]', '', regex=True).astype(float)

    # Drop arbitrary rows
    ds['Roll Number'] = ds['Roll Number'].astype(str)
    roll_nos_to_drop = ['0107376627423 49.80160863413996', '9257376']
    ds = ds[~ds['Roll Number'].isin(roll_nos_to_drop)]

    # Drop unnamed index column
    if ds.columns[0].startswith('Unnamed'):
        ds = ds.iloc[:, 1:]

    # Impute missing values
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
                 'Full Address','Geometry','Detail URL','Assessed Value 1','Assessed Value 2','Assessed Value 3',
                 'Assessed Value 4','Assessed Value 5','Property Class 1','Property Class 2','Property Class 3',
                 'Property Class 4','Property Class 5','Status 1','Status 2','Status 3','Status 4','Status 5',
                 'Roll Number','Unit Number','Street Name','Street Suffix','Dwelling Units','Multiple Residences',
                 'Property Influences','Number Floors (Condo)']
    ds.drop(columns=[col for col in drop_cols if col in ds.columns], inplace=True)

    # Encode categorical columns
    categorical_cols = ds.select_dtypes(include=['object', 'bool']).columns.tolist()
    category_counts = ds[categorical_cols].nunique()
    low_cardinality = category_counts[category_counts <= 20].index.tolist()
    high_cardinality = category_counts[category_counts > 20].index.tolist()
    df_encoded = pd.get_dummies(ds, columns=low_cardinality, drop_first=True)
    for col in high_cardinality:
        freq_encoding = ds[col].value_counts()
        df_encoded[col + '_freq'] = ds[col].map(freq_encoding)
    df_encoded.drop(columns=high_cardinality, inplace=True)

    # Modeling
    df_encoded.dropna(subset=["Total Assessed Value"], inplace=True)
    X = df_encoded.drop(columns=["Total Assessed Value"])
    y = df_encoded["Total Assessed Value"]

    scalers = {"MinMax": MinMaxScaler(), "ZScore": StandardScaler()}
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(),
        "Linear Regression": LinearRegression()
    }

    performance = []
    model_objects = {}
    for scale_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (y.max() - y.min())
            performance.append({
                "Model": f"{model_name} ({scale_name})",
                "R²": round(r2, 3),
                "MAPE": round(mape, 3),
                "nRMSE": round(nrmse, 3)
            })
            model_objects[f"{model_name} ({scale_name})"] = (model, scaler)

    performance_df = pd.DataFrame(performance)
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(performance_df)

    # Select best model based on R² and nRMSE
    best_model_row = performance_df.sort_values(by=["R²", "nRMSE"], ascending=[False, True]).iloc[0]
    best_model_name = best_model_row["Model"]
    best_model, best_scaler = model_objects[best_model_name]

    # Prediction UI
    st.subheader("🏡 Predict Property Valuation")
    col1, col2 = st.columns(2)
    year_built = col1.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    total_living_area = col2.number_input("Total Living Area (sq ft)", min_value=100, max_value=10000, value=1500)
    rooms = col1.number_input("Number of Rooms", min_value=1, max_value=20, value=5)
    bathrooms = col2.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    neighbourhoods = ds["Neighbourhood Area"].dropna().unique().tolist()
    selected_neighbourhood = st.selectbox("Neighbourhood Area", sorted(neighbourhoods))
    neighbourhood_freq = ds["Neighbourhood Area"].value_counts().get(selected_neighbourhood, 0)

    # Prepare input for prediction
    input_dict = {
        "Year Built": year_built,
        "Total Living Area": total_living_area,
        "Rooms": rooms,
        "Bathrooms": bathrooms,
        "Neighbourhood Area_freq": neighbourhood_freq
    }

    # Fill missing features with median or zero
    for feature in X.columns:
        if feature not in input_dict:
            input_dict[feature] = X[feature].median() if feature in X.columns else 0

    input_df = pd.DataFrame([input_dict])
    input_scaled = best_scaler.transform(input_df[X.columns])

if st.button("Predict"):
    predicted_value = best_model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Assessed Value: ${predicted_value:,.2f}")
    st.success(f"💰 Predicted Assessed Value: ${predicted_value:,.2f}")

