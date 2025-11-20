import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
import os

st.set_page_config(page_title='Lab5 - RNN Demo', layout='wide')

st.title('Lab 5 — RNN Time-Series Demo')
st.markdown('Simple demo for the RNN model trained on `HistoricalQuotes.csv`.')

DATA_PATH = 'HistoricalQuotes.csv'
MODEL_PATH = 'models/lab5_rnn.h5'


@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    return df


def detect_and_clean_close(df):
    # Parse date
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    # Find close column
    close_col = next((c for c in df.columns if 'close' in c.lower()), None)
    if close_col is None:
        st.error("No column containing 'close' found in the CSV.")
        return None, None

    # Clean and convert
    df['Close'] = df[close_col].astype(str).str.replace(
        r'[\$,]', '', regex=True).str.strip().astype(float)
    return df, close_col


def create_sequences(series, seq_length):
    arr = series.values if hasattr(series, 'values') else np.array(series)
    X, y = [], []
    for i in range(len(arr) - seq_length):
        X.append(arr[i: i + seq_length])
        y.append(arr[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


@st.cache_resource
def build_rnn_model(sequence_length):
    model = Sequential([
        SimpleRNN(50, input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Sidebar controls
st.sidebar.header('Controls')
use_local = st.sidebar.checkbox('Use local CSV file', value=True)
sequence_length = st.sidebar.slider('Sequence length (days)', 30, 120, 60)
train_epochs = st.sidebar.slider('Training epochs (if training)', 1, 100, 10)
train_button = st.sidebar.button('Train model')
load_button = st.sidebar.button('Load saved model')
save_button = st.sidebar.button('Save current model')

# Load data
if use_local and os.path.exists(DATA_PATH):
    df = load_csv(DATA_PATH)
else:
    uploaded = st.file_uploader('Upload HistoricalQuotes.csv', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning(
            'Please upload `HistoricalQuotes.csv` or enable `Use local CSV file`.')
        st.stop()

# Clean and prepare
df, close_col = detect_and_clean_close(df)
if df is None:
    st.stop()

st.subheader('Dataset preview')
st.write(df.head())

st.subheader('Close price over time')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Date'], df['Close'], lw=1)
ax.set_xlabel('Date')
ax.set_ylabel('Close')
plt.xticks(rotation=45)
st.pyplot(fig)

# Scale close
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close_scaled'] = scaler.fit_transform(df[['Close']])

# Chronological split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

st.write(
    f"Total samples: {len(df)}, train: {len(train_df)}, test: {len(test_df)}")

# Create sequences
X_train, y_train = create_sequences(train_df['Close_scaled'], sequence_length)
X_test, y_test = create_sequences(test_df['Close_scaled'], sequence_length)

st.write('Sequence shapes:')
st.write({'X_train': X_train.shape, 'y_train': y_train.shape,
         'X_test': X_test.shape, 'y_test': y_test.shape})

# Build or load model
model = None
if load_button and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success(f'Loaded model from {MODEL_PATH}')
    except Exception as e:
        st.error(f'Error loading model: {e}')

if train_button:
    st.info('Training model...')
    model = build_rnn_model(sequence_length)
    history = model.fit(X_train, y_train, epochs=train_epochs,
                        batch_size=32, validation_data=(X_test, y_test), verbose=0)
    st.success('Training finished')
    # Show loss plot
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='train_loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.legend()
    st.pyplot(fig2)

# If model available, run predictions
if model is not None:
    st.subheader('Model predictions on test set')
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Prepare dates for plotting (skip first sequence_length rows of test_df)
    test_dates = test_df['Date'].iloc[sequence_length:].reset_index(drop=True)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(test_dates, y_test_orig, label='Actual Close', lw=1)
    ax3.plot(test_dates, y_pred, label='Predicted Close', lw=1)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Close')
    ax3.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred)) if 'mean_squared_error' in globals(
    ) else np.sqrt(((y_test_orig - y_pred)**2).mean())
    st.write({'MAE': float(mae), 'RMSE': float(rmse)})

    if save_button:
        os.makedirs('models', exist_ok=True)
        model.save(MODEL_PATH)
        st.success(f'Model saved to {MODEL_PATH}')

else:
    st.info(
        'No model available. Train a model or load a saved model to see predictions.')

st.markdown('---')