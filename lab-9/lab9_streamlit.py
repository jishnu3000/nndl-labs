import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import os

st.set_page_config(
    page_title='Lab9 — Temperature Forecasting RNN', layout='wide')
st.title('Lab 9 — Temperature Forecasting with RNN Demo')
st.markdown(
    'Interactive demo for temperature time series forecasting using LSTM encoder-decoder.')

# Paths and configuration
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'temperature_forecast_rnn.keras')

# Functions from notebook


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess both training and testing datasets"""
    try:
        # Load datasets with appropriate separators
        train = pd.read_csv('training.csv', sep='\t')
        test = pd.read_csv('testing.csv', sep=',')

        # Clean and preprocess train set
        train['Date time'] = pd.to_datetime(train['Date time'])
        train['Temperature'] = pd.to_numeric(
            train['Temperature'], errors='coerce')
        train = train.dropna(subset=['Temperature'])
        train = train.sort_values('Date time')

        # Clean and preprocess test set
        test['Date time'] = pd.to_datetime(test['Date time'])
        test['Temperature'] = pd.to_numeric(
            test['Temperature'], errors='coerce')
        test = test.dropna(subset=['Temperature'])
        test = test.sort_values('Date time')

        return train, test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_resource
def build_rnn_model(sequence_length=10):
    """Build the LSTM encoder-decoder model"""
    # Encoder
    encoder_inputs = layers.Input(shape=(sequence_length, 1))
    encoder_lstm = layers.LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.RepeatVector(1)(encoder_outputs)
    decoder_lstm = layers.LSTM(64, return_sequences=False, return_state=False)
    decoder_outputs = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(1, activation='linear')
    outputs = decoder_dense(decoder_outputs)

    # Model
    model = models.Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# Sidebar controls
st.sidebar.header('Configuration')
sequence_length = st.sidebar.slider('Sequence length', 5, 20, 10)
epochs = st.sidebar.slider('Training epochs', 10, 100, 50)
batch_size = st.sidebar.selectbox('Batch size', [16, 32, 64], index=1)

# Model operations
train_button = st.sidebar.button('Train Model')
load_button = st.sidebar.button('Load Saved Model')
save_button = st.sidebar.button('Save Current Model')

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# Auto-load saved model
if st.session_state.model is None and os.path.exists(MODEL_PATH):
    try:
        st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success('🎯 Saved model loaded automatically!')
    except Exception as e:
        st.sidebar.warning(f'Could not auto-load model: {e}')

# Load data
train_data, test_data = load_and_preprocess_data()
if train_data is not None and test_data is not None:
    st.session_state.train_data = train_data
    st.session_state.test_data = test_data

    # Show data info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Training Data')
        st.write(f"Shape: {train_data.shape}")
        st.write(
            f"Date range: {train_data['Date time'].min()} to {train_data['Date time'].max()}")
        st.write(train_data.head())

    with col2:
        st.subheader('Testing Data')
        st.write(f"Shape: {test_data.shape}")
        st.write(
            f"Date range: {test_data['Date time'].min()} to {test_data['Date time'].max()}")
        st.write(test_data.head())

# Model operations
if load_button:
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
            st.success('Model loaded successfully!')
        except Exception as e:
            st.error(f'Error loading model: {e}')
    else:
        st.info('No saved model found. Train a model first.')

if train_button and st.session_state.train_data is not None:
    with st.spinner('Training model...'):
        # Prepare data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(
            st.session_state.train_data[['Temperature']])
        test_scaled = scaler.transform(
            st.session_state.test_data[['Temperature']])

        # Store scaler
        st.session_state.scaler = scaler

        # Create sequences
        X_train, y_train = create_sequences(
            train_scaled.flatten(), sequence_length)
        X_test, y_test = create_sequences(
            test_scaled.flatten(), sequence_length)

        # Reshape for RNN
        X_train_rnn = X_train[..., np.newaxis]
        X_test_rnn = X_test[..., np.newaxis]

        # Build and train model
        st.session_state.model = build_rnn_model(sequence_length)

        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Custom callback for progress
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f'Epoch {epoch + 1}/{epochs} - Loss: {logs.get("loss", 0):.4f}')

        history = st.session_state.model.fit(
            X_train_rnn, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[ProgressCallback()]
        )

        progress_bar.empty()
        status_text.empty()

        # Plot training curves
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Progress')
        ax.legend()
        st.pyplot(fig)

        st.success('Training completed!')

if save_button and st.session_state.model is not None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    st.session_state.model.save(MODEL_PATH)
    st.success('Model saved successfully!')

# Prediction and visualization
if st.session_state.model is not None and st.session_state.train_data is not None:
    st.header('Temperature Forecasting Results')

    # Prepare test data for prediction
    if st.session_state.scaler is None:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(
            st.session_state.train_data[['Temperature']])
        test_scaled = scaler.transform(
            st.session_state.test_data[['Temperature']])
        st.session_state.scaler = scaler
    else:
        scaler = st.session_state.scaler
        test_scaled = scaler.transform(
            st.session_state.test_data[['Temperature']])

    # Create test sequences
    X_test, y_test = create_sequences(test_scaled.flatten(), sequence_length)
    X_test_rnn = X_test[..., np.newaxis]

    # Make predictions
    predictions = st.session_state.model.predict(X_test_rnn, verbose=0)

    # Plot predictions vs actual
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Full comparison
    ax1.plot(y_test, label='Actual Temperature (Scaled)', alpha=0.7)
    ax1.plot(predictions.flatten(),
             label='Predicted Temperature (Scaled)', alpha=0.7)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Scaled Temperature')
    ax1.set_title('Temperature Forecasting: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoomed view (first 100 points)
    n_zoom = min(100, len(y_test))
    ax2.plot(y_test[:n_zoom], label='Actual Temperature (Scaled)',
             marker='o', markersize=3)
    ax2.plot(predictions.flatten()[
             :n_zoom], label='Predicted Temperature (Scaled)', marker='s', markersize=3)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Scaled Temperature')
    ax2.set_title(f'Temperature Forecasting (First {n_zoom} predictions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate metrics
    mse = np.mean((y_test - predictions.flatten()) ** 2)
    mae = np.mean(np.abs(y_test - predictions.flatten()))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Mean Squared Error', f'{mse:.6f}')
    with col2:
        st.metric('Mean Absolute Error', f'{mae:.6f}')
    with col3:
        st.metric('Test Samples', len(y_test))

# Instructions
if st.session_state.model is None:
    st.info('👆 Use the sidebar to train a new model or load a saved one.')

st.markdown('---')
