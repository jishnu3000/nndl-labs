import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img

st.set_page_config(page_title='Lab8 — Insect Autoencoder Demo', layout='wide')
st.title('Lab 8 — Insect Image Autoencoder Demo')
st.markdown(
    'Interactive demo for insect image reconstruction using CNN autoencoder.')

# Paths and configuration
IMG_SIZE = (128, 128)
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'insect_autoencoder.keras')


@st.cache_resource
def build_autoencoder(img_size=(128, 128)):
    """Build the CNN autoencoder architecture"""
    # Encoder
    encoder_input = layers.Input(shape=img_size + (3,))
    x = layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(
        128, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2DTranspose(
        64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(encoder_input, decoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def preprocess_image(uploaded_file, target_size=(128, 128)):
    """Preprocess uploaded image for the model"""
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return img_array, image


def load_sample_data():
    """Load sample insect images from local data folder"""
    data_path = 'data'  # Local data folder

    if not os.path.exists(data_path):
        st.warning(
            f"Local data folder '{data_path}' not found. Please ensure the data folder exists with insect categories.")
        return None, None

    try:
        sample_images = []
        sample_labels = []

        # Get available categories from data folder
        categories = [d for d in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, d))]

        if not categories:
            st.warning("No category folders found in data directory.")
            return None, None

        # Load a few sample images from each category
        for label in categories[:5]:  # Limit to first 5 categories
            label_dir = os.path.join(data_path, label)
            img_files = [f for f in os.listdir(
                label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_file in img_files[:3]:  # 3 images per category
                try:
                    img_path = os.path.join(label_dir, img_file)
                    img = load_img(img_path, target_size=IMG_SIZE)
                    img_array = img_to_array(img) / 255.0
                    sample_images.append(img_array)
                    sample_labels.append(label)
                except Exception:
                    continue

        if len(sample_images) == 0:
            st.warning("No valid images found in data folder.")
            return None, None

        return np.array(sample_images), sample_labels
    except Exception as e:
        st.error(f"Could not load sample data: {e}")
        return None, None


# Sidebar controls
st.sidebar.header('Controls')

# Show data folder status
data_path = 'data'
if os.path.exists(data_path):
    categories = [d for d in os.listdir(
        data_path) if os.path.isdir(os.path.join(data_path, d))]
    st.sidebar.info(f'📁 Local data: {len(categories)} categories found')
else:
    st.sidebar.warning('📁 No local data folder found')

use_sample_data = st.sidebar.checkbox('Use sample dataset', value=True)
train_epochs = st.sidebar.slider('Training epochs', 1, 50, 10)
batch_size = st.sidebar.selectbox('Batch size', [16, 32, 64], index=1)

# Model operations
train_button = st.sidebar.button('Train Model')
load_button = st.sidebar.button('Load Saved Model')
save_button = st.sidebar.button('Save Current Model')

# Initialize session state for model and try to load saved model
if 'model' not in st.session_state:
    st.session_state.model = None
    # Try to automatically load saved model on startup
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
            st.sidebar.success('🎯 Saved model loaded automatically!')
        except Exception as e:
            st.sidebar.warning(f'Could not auto-load model: {e}')

# Load model
if load_button:
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
            st.success('Model loaded successfully!')
        except Exception as e:
            st.error(f'Error loading model: {e}')
    else:
        st.info('No saved model found. Train a model first.')

# Train model
if train_button:
    with st.spinner('Training model...'):
        if use_sample_data:
            sample_images, sample_labels = load_sample_data()
            if sample_images is not None:
                st.session_state.model = build_autoencoder(IMG_SIZE)

                # Simple train/validation split
                split_idx = int(0.8 * len(sample_images))
                X_train = sample_images[:split_idx]
                X_val = sample_images[split_idx:]

                history = st.session_state.model.fit(
                    X_train, X_train,
                    epochs=train_epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, X_val),
                    verbose=0
                )

                # Plot training curves
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('Training Progress')
                ax.legend()
                st.pyplot(fig)

                st.success('Training completed!')
            else:
                st.error('Could not load sample data for training')
        else:
            st.warning(
                'Please enable "Use sample dataset" or upload training images')

# Save model
if save_button and st.session_state.model is not None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    st.session_state.model.save(MODEL_PATH)
    st.success('Model saved successfully!')

# Main interface
st.header('Image Reconstruction Demo')

# Image upload section
uploaded_file = st.file_uploader(
    "Upload an insect image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None and st.session_state.model is not None:
    # Preprocess and predict
    img_array, original_img = preprocess_image(uploaded_file, IMG_SIZE)
    img_batch = np.expand_dims(img_array, axis=0)

    # Reconstruct
    reconstructed = st.session_state.model.predict(img_batch, verbose=0)
    reconstructed_img = np.clip(reconstructed[0], 0, 1)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Original Image')
        st.image(original_img, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.subheader('Reconstructed Image')
        st.image(reconstructed_img, caption='Autoencoder Output',
                 use_container_width=True)

    # Calculate and display reconstruction error
    mse = np.mean((img_array - reconstructed_img) ** 2)
    st.metric('Reconstruction MSE', f'{mse:.6f}')

# Sample gallery section
elif use_sample_data and st.session_state.model is not None:
    st.subheader('Sample Reconstructions')
    sample_images, sample_labels = load_sample_data()

    if sample_images is not None:
        # Select a few samples for display
        n_samples = min(6, len(sample_images))
        sample_batch = sample_images[:n_samples]
        reconstructed_batch = st.session_state.model.predict(
            sample_batch, verbose=0)

        # Display in grid
        cols = st.columns(n_samples)
        for i in range(n_samples):
            with cols[i]:
                st.write(
                    f"**{sample_labels[i] if i < len(sample_labels) else 'Unknown'}**")
                st.image(sample_batch[i],
                         caption='Original', use_container_width=True)
                st.image(np.clip(
                    reconstructed_batch[i], 0, 1), caption='Reconstructed', use_container_width=True)
                mse = np.mean((sample_batch[i] - reconstructed_batch[i]) ** 2)
                st.write(f"MSE: {mse:.6f}")

# Instructions
if st.session_state.model is None:
    st.info('👆 Use the sidebar to train a new model or load a saved one.')

st.markdown('---')
