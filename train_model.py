# train_model.py

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

DATA_DIR = 'data'
MODEL_DIR = 'models'
SEQ_LEN = 60
EPOCHS = 20
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i - seq_len:i])
        y.append(data[i, 3])  # 🔄 Predicting 'Close' price, which is index 3 after scaling ['Open', 'High', 'Low', 'Close', 'Volume']
    return np.array(x), np.array(y)

def train_for_ticker(ticker, filepath):
    print(f"📈 Training model for {ticker}...")

    df = pd.read_csv(filepath, index_col=0)
    df.dropna(inplace=True)

    # ✅ Use all relevant features
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        print(f"⚠️ Skipping {ticker}: Missing one or more required columns {required_columns}")
        return

    data = df[required_columns]  # Keep as DataFrame with column names

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)  # Uses DataFrame (no warning)


    # Create sequences
    x, y = create_sequences(scaled_data, SEQ_LEN)
    if len(x) == 0:
        print(f"⚠️ Not enough data to create sequences for {ticker}.")
        return

    # Train-test split
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    # 🔄 Updated input shape to match (SEQ_LEN, 5 features)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, x.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test), callbacks=[early_stop], verbose=0)

    # Save model and scaler
    model.save(os.path.join(MODEL_DIR, f"{ticker}_model.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"))
    print(f"✅ Model and scaler saved for {ticker}")

def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        print("⚠️ No CSV files found in data/")
        return

    for file in files:
        ticker = file.replace('.csv', '')
        filepath = os.path.join(DATA_DIR, file)
        train_for_ticker(ticker, filepath)

if __name__ == "__main__":
    main()
