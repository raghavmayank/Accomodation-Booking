# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Load models
with open('./revenue_forcast.pkl', 'rb') as file:
    arima_model = pickle.load(file)

# Load data
file_path = './Dataset/hotel_booking.csv'
df = pd.read_csv(file_path)

# Preprocess data for Streamlit
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Streamlit app
st.title('Hotel Booking Analysis')

# Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', ['Overview', 'Revenue Forecasting', 'Predict Booking Cancellations', 'Market Segmentation', 'Customer Lifetime Value'])

if options == 'Overview':
    st.header('Overview')
    st.write('This app provides insights and predictions for hotel bookings.')

elif options == 'Revenue Forecasting':
    # Streamlit app title
    st.title('Hotel Booking Revenue Forecasting with SARIMA')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the dataset
        st.write("## Dataset Preview")
        st.write(data.head())
        
        # Convert arrival_date_year and arrival_date_month to a datetime format
        data['arrival_date'] = pd.to_datetime(data['arrival_date_year'].astype(str) + '-' + 
                                              data['arrival_date_month'].astype(str) + '-01')
        data['arrival_date'] += MonthEnd(0)
        
        # Calculate monthly revenue
        monthly_revenue = data[data['is_canceled'] == 0].groupby('arrival_date')['adr'].sum().reset_index()
        
        # Plot monthly revenue
        st.write("## Monthly Revenue")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='arrival_date', y='adr', data=monthly_revenue)
        plt.title('Monthly Revenue')
        plt.xlabel('Month')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        # Check for stationarity
        result = adfuller(monthly_revenue['adr'])
        st.write(f'## ADF Statistic: {result[0]}')
        st.write(f'## p-value: {result[1]}')

        # If the series is not stationary, take the first difference
        monthly_revenue['adr_diff'] = monthly_revenue['adr'].diff().dropna()

        # Model parameters
        p = st.slider('AR order (p)', 0, 5, 1)
        d = st.slider('Differencing order (d)', 0, 2, 1)
        q = st.slider('MA order (q)', 0, 5, 1)
        P = st.slider('Seasonal AR order (P)', 0, 2, 1)
        D = st.slider('Seasonal differencing order (D)', 0, 2, 1)
        Q = st.slider('Seasonal MA order (Q)', 0, 2, 1)

        # Fit the SARIMA model with user-defined parameters
        model = SARIMAX(monthly_revenue['adr'], 
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, 12))
        model_fit = model.fit(disp=False)

        # Make predictions
        forecast_steps = 12  # Forecast for the next 12 months
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=monthly_revenue['arrival_date'].max(), 
                                       periods=forecast_steps, freq='M')

        forecast_df = pd.DataFrame({'arrival_date': forecast_index, 
                                    'forecast': forecast.predicted_mean})

        # Plot the results
        st.write("## Revenue Forecast")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='arrival_date', y='adr', data=monthly_revenue, label='Historical Revenue')
        sns.lineplot(x='arrival_date', y='forecast', data=forecast_df, label='Forecasted Revenue')
        plt.title('Revenue Forecast')
        plt.xlabel('Month')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display forecasted values
        st.write("## Forecasted Revenue for the Next 12 Months")
        st.write(forecast_df.set_index('arrival_date'))


elif options == 'Predict Booking Cancellations':
    st.header('Predict Booking Cancellations')
    st.write('Provide input data to predict if a booking will be canceled.')

    input_data = {}
    for col in df.drop(columns=['is_canceled']).columns:
        input_data[col] = st.text_input(f'{col}:', value='0')

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = random_forest_model.predict(input_df)
    st.write('Prediction:', 'Canceled' if prediction[0] else 'Not Canceled')

elif options == 'Market Segmentation':
    st.header('Market Segmentation')
    segmentation_features = df[['total_guests', 'total_of_special_requests', 'lead_time', 'is_repeated_guest']]
    scaler = StandardScaler()
    segmentation_features_scaled = scaler.fit_transform(segmentation_features)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['customer_segment'] = kmeans.fit_predict(segmentation_features_scaled)
    
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=segmentation_features_scaled[:, 0], y=segmentation_features_scaled[:, 1], hue=df['customer_segment'], palette='viridis')
    plt.title('Customer Segmentation')
    plt.xlabel('Total Guests (Standardized)')
    plt.ylabel('Total Special Requests (Standardized)')
    st.pyplot(plt)

elif options == 'Customer Lifetime Value':
    st.header('Customer Lifetime Value')
    clv_df = df.groupby('customer_id')['revenue'].sum().reset_index()
    clv_df.columns = ['customer_id', 'lifetime_value']
    
    plt.figure(figsize=(10, 5))
    sns.histplot(clv_df['lifetime_value'], kde=True)
    plt.title('Customer Lifetime Value Distribution')
    plt.xlabel('Lifetime Value')
    plt.ylabel('Frequency')
    st.pyplot(plt)
