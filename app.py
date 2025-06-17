import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from predict import predict_next_day_price
from tensorflow.keras.models import load_model
import joblib
import os

# Constants
TICKERS = ['AAPL', 'TSLA', 'JPM', 'JNJ', 'XOM']
LOOKBACK = 60

# --- Page Title ---
st.title("📈 Stock Price Predictor")
st.markdown("Predicts the **next-day Close price** using an LSTM model.")

# --- User selects a stock ---
selected_ticker = st.selectbox("Choose a Stock Ticker:", TICKERS)

# --- Prediction Button ---
if st.button("🔮 Predict Next Day Price"):
    pred_price, df_recent, pred_series = predict_next_day_price(selected_ticker)

    # Show prediction
    st.metric(label=f"Next Day Predicted Close Price for {selected_ticker}", value=f"${pred_price:.2f}")

    # Plot recent actual + predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_recent['Close'].values, name='Actual Close Price', mode='lines+markers'))
    fig.add_trace(go.Scatter(y=pred_series, name='Predicted Price (1-day ahead)', mode='lines+markers'))
    fig.update_layout(title=f'{selected_ticker} - Last {LOOKBACK} Days + 1 Predicted', xaxis_title='Days', yaxis_title='Price ($)')
    st.plotly_chart(fig)
