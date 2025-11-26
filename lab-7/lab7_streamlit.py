import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Lab7 Demo — Autoencoders', layout='wide')
st.title('Lab 7 — Autoencoder')

# Paths
MODEL_DIR = 'models'
CIFAR_MODEL = 'cnn_autoencoder_cifar10.keras'
LSTM_MODEL = 'lstm_autoencoder_sinewaves.keras'

# --------------------------------------------------
# CIFAR Autoencoder Demo
# --------------------------------------------------
st.header('CIFAR-10 CNN Autoencoder')
col1, col2 = st.columns([1, 1])
with col1:
    st.write('Configure training (keeps small for demo)')
    cifar_epochs = st.number_input(
        'Epochs', min_value=1, max_value=20, value=5)
    cifar_batch = st.selectbox('Batch size', [64, 128, 256], index=1)
    cifar_train = st.button('Train CIFAR Autoencoder')
    cifar_load = st.button('Load saved CIFAR model')
with col2:
    st.write('Preview & outputs')


@st.cache_data
def load_cifar_subset(n_samples=2000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    X = np.concatenate([x_train, x_test])
    X = X.astype('float32') / 255.0
    # use a subset for quick demo
    if X.shape[0] > n_samples:
        idx = np.random.RandomState(42).choice(
            X.shape[0], n_samples, replace=False)
        X = X[idx]
    return X


@st.cache_resource
def build_cifar_autoencoder():
    input_img = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      activation='relu')(input_img)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)

    x = layers.Dense(4*4*128, activation='relu')(latent)
    x = layers.Reshape((4, 4, 128))(x)
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding='same', activation='relu')(x)
    decoded = layers.Conv2DTranspose(
        3, 3, strides=2, padding='same', activation='sigmoid')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# CIFAR operations
cifar_msg = st.empty()
if cifar_load:
    if os.path.exists(CIFAR_MODEL):
        try:
            caf_model = keras.models.load_model(CIFAR_MODEL)
            cifar_msg.success('Loaded CIFAR autoencoder model')
        except Exception as e:
            cifar_msg.error(f'Failed to load model: {e}')
    else:
        cifar_msg.info('No saved CIFAR model found. Train and save first.')

if cifar_train:
    X = load_cifar_subset()
    auto = build_cifar_autoencoder()
    st.info('Training CIFAR autoencoder on a small subset — this may take a minute')
    history = auto.fit(X, X, epochs=int(cifar_epochs),
                       batch_size=int(cifar_batch), verbose=1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    auto.save(CIFAR_MODEL)
    cifar_msg.success('Training complete and model saved.')

# If model exists, show reconstructions
if os.path.exists(CIFAR_MODEL):
    try:
        model = keras.models.load_model(CIFAR_MODEL)
        Xvis = load_cifar_subset(n_samples=10)
        recon = model.predict(Xvis)
        st.subheader('Sample reconstructions')
        fig, axes = plt.subplots(2, 10, figsize=(18, 4))
        for i in range(10):
            axes[0, i].imshow(Xvis[i])
            axes[0, i].axis('off')
            axes[1, i].imshow(np.clip(recon[i], 0, 1))
            axes[1, i].axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Error showing reconstructions: {e}')

st.markdown('---')

# --------------------------------------------------
# LSTM Autoencoder Demo (synthetic sequences)
# --------------------------------------------------
st.header('LSTM Autoencoder — Synthetic Sine Sequences')
col1, col2 = st.columns([1, 1])
with col1:
    seq_train = st.button('Train LSTM Autoencoder')
    seq_load = st.button('Load saved LSTM model')
    seq_epochs = st.number_input(
        'Epochs (LSTM)', min_value=1, max_value=100, value=10)
    seq_batch = st.selectbox('Batch size (LSTM)', [32, 64, 128], index=0)
with col2:
    st.write('Preview & outputs')


@st.cache_data
def make_sine_sequences(num_samples=1000, timesteps=30):
    t = np.linspace(0, 4 * np.pi, timesteps)
    X = np.array([np.sin(t + np.random.uniform(-0.5, 0.5))
                 for _ in range(num_samples)])
    return X[..., np.newaxis].astype('float32')


@st.cache_resource
def build_lstm_autoencoder(timesteps=30, features=1, latent_dim=32):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(64, return_sequences=True)(inputs)
    encoded = LSTM(latent_dim)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, return_sequences=True)(decoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(features))(decoded)
    auto = Model(inputs, outputs)
    auto.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return auto


seq_msg = st.empty()
if seq_load:
    if os.path.exists(LSTM_MODEL):
        try:
            lstm_model = keras.models.load_model(LSTM_MODEL)
            seq_msg.success('Loaded LSTM autoencoder model')
        except Exception as e:
            seq_msg.error(f'Failed to load LSTM model: {e}')
    else:
        seq_msg.info('No saved LSTM model found. Train and save first.')

if seq_train:
    st.info('Preparing synthetic sequences and training LSTM autoencoder')
    Xseq = make_sine_sequences()
    Xtr, Xval = train_test_split(Xseq, test_size=0.2, random_state=42)
    lstm = build_lstm_autoencoder(
        timesteps=Xseq.shape[1], features=1, latent_dim=32)
    h = lstm.fit(Xtr, Xtr, validation_data=(Xval, Xval), epochs=int(
        seq_epochs), batch_size=int(seq_batch), verbose=1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    lstm.save(LSTM_MODEL)
    seq_msg.success('LSTM training complete and model saved.')
    # Plot a few reconstructions
    pred = lstm.predict(Xval[:5])
    fig, ax = plt.subplots(5, 1, figsize=(8, 8))
    for i in range(5):
        ax[i].plot(Xval[i].squeeze(), label='Original')
        ax[i].plot(pred[i].squeeze(), label='Reconstructed')
        ax[i].legend()
    st.pyplot(fig)

# If LSTM model exists, show sample reconstructions
if os.path.exists(LSTM_MODEL):
    try:
        lm = keras.models.load_model(LSTM_MODEL)
        Xs = make_sine_sequences(num_samples=5)
        preds = lm.predict(Xs)
        st.subheader('LSTM reconstructions (sample)')
        fig, ax = plt.subplots(5, 1, figsize=(8, 8))
        for i in range(5):
            ax[i].plot(Xs[i].squeeze(), label='Original')
            ax[i].plot(preds[i].squeeze(), label='Reconstructed')
            ax[i].legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Error showing LSTM reconstructions: {e}')

st.markdown('---')
st.write('Notes: This demo uses small subsets and low epochs to run quickly for interactive use. For serious training, increase dataset sizes and epochs, and run in a GPU-enabled environment.')
