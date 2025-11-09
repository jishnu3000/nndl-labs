import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="CNN Image Classification Demo",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ CNN Image Classification Demo")
st.markdown("---")
st.markdown(
    "This app demonstrates the CNN models trained for image classification using TensorFlow/Keras.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Model Overview", "Image Classification",
        "Model Comparison", "Technical Details"]
)

# Class names (based on Intel Image Classification dataset)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def get_class_labels(num_classes):
    """Get class labels based on number of classes"""
    if num_classes == len(CLASS_NAMES):
        return CLASS_NAMES
    elif num_classes < len(CLASS_NAMES):
        return CLASS_NAMES[:num_classes]
    else:
        # If model has more classes than expected, generate generic labels
        return [f"Class_{i}" for i in range(num_classes)]


@st.cache_resource
def load_model(model_path):
    """Load a trained model with caching"""
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image, target_size=(150, 150)):
    """Preprocess uploaded image for model prediction"""
    # Resize image
    image = image.resize(target_size)

    # Convert to array and normalize
    img_array = np.array(image)

    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Add batch dimension and normalize
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0

    return img_array


def predict_image(model, image):
    """Make prediction on preprocessed image"""
    try:
        prediction = model.predict(image, verbose=0)

        # Ensure prediction is in the right format
        if len(prediction.shape) > 1:
            prediction_probs = prediction[0]
        else:
            prediction_probs = prediction

        predicted_class = np.argmax(prediction_probs)
        confidence = np.max(prediction_probs)

        return predicted_class, confidence, prediction_probs
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None


# Page 1: Model Overview
if page == "Model Overview":
    st.header("📊 Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Information")
        st.info(f"""
        **Dataset**: Intel Image Classification
        
        **Classes**: {len(CLASS_NAMES)}
        - {', '.join(CLASS_NAMES)}
        
        **Image Size**: 150x150 pixels
        
        **Architecture**: Convolutional Neural Network (CNN)
        """)

    with col2:
        st.subheader("Available Models")

        # Check which models are available
        models_dir = "models"
        available_models = []

        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(
                models_dir) if f.endswith('.h5')]
            for model_file in model_files:
                file_path = os.path.join(models_dir, model_file)
                file_size = os.path.getsize(
                    file_path) / (1024 * 1024)  # Size in MB
                available_models.append(f"✅ {model_file} ({file_size:.1f} MB)")

        if available_models:
            for model in available_models:
                st.success(model)
        else:
            st.warning(
                "No trained models found. Please run the training notebook first.")

    st.subheader("Model Architecture")
    st.code("""
    CNN Architecture:
    
    1. Data Preprocessing
       - Rescaling (1./255)
       - Data Augmentation (for improved model)
    
    2. Convolutional Blocks
       - Conv2D layers (32, 64, 128/256 filters)
       - Batch Normalization
       - MaxPooling2D
       - Dropout (0.25)
    
    3. Dense Layers
       - Flatten
       - Dense (256/512 neurons)
       - Batch Normalization
       - Dropout (0.5)
       - Output Dense (6 classes, softmax)
    """, language="text")

# Page 2: Image Classification
elif page == "Image Classification":
    st.header("🔍 Image Classification")

    # Model selection
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Model")
        model_choice = st.selectbox(
            "Choose a trained model:",
            ["best_improved_model.h5", "best_cnn_model.h5",
                "final_improved_model.h5", "final_cnn_model.h5"]
        )

        # Load selected model
        model_path = f"models/{model_choice}"
        model = load_model(model_path)

        if model:
            st.success(f"✅ Model loaded: {model_choice}")
        else:
            st.error(f"❌ Could not load: {model_choice}")

    with col2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to classify"
        )

    if uploaded_file is not None and model is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        with st.spinner("Classifying image..."):
            processed_image = preprocess_image(image)
            predicted_class, confidence, all_predictions = predict_image(
                model, processed_image)

        if predicted_class is not None:
            # Display results
            predicted_class_name = CLASS_NAMES[predicted_class] if predicted_class < len(
                CLASS_NAMES) else f"Class_{predicted_class}"
            st.success(
                f"**Prediction**: {predicted_class_name} ({confidence:.2%} confidence)")

            # Show all class probabilities
            st.subheader("Prediction Probabilities")

            # Create DataFrame for better visualization - handle length mismatch
            num_predictions = len(all_predictions)
            class_labels = get_class_labels(num_predictions)

            prob_df = pd.DataFrame({
                'Class': class_labels,
                'Probability': all_predictions
            }).sort_values('Probability', ascending=False)

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(prob_df['Class'], prob_df['Probability'])

            # Highlight the predicted class
            bars[prob_df.index[0]].set_color('lightgreen')

            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            # Show detailed probabilities
            st.subheader("Detailed Results")

            # Debug information
            st.write(
                f"Model output shape: {all_predictions.shape if hasattr(all_predictions, 'shape') else len(all_predictions)}")
            st.write(f"Number of classes expected: {len(CLASS_NAMES)}")
            st.write(f"Number of predictions received: {len(all_predictions)}")

            for i, (class_name, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
                if i == 0:  # Highest probability
                    st.success(f"🥇 **{class_name}**: {prob:.4f} ({prob:.2%})")
                elif i == 1:  # Second highest
                    st.info(f"🥈 {class_name}: {prob:.4f} ({prob:.2%})")
                else:
                    st.write(f"• {class_name}: {prob:.4f} ({prob:.2%})")

# Page 3: Model Comparison
elif page == "Model Comparison":
    st.header("⚖️ Model Comparison")

    # Load models for comparison
    basic_model = load_model("models/best_cnn_model.h5")
    improved_model = load_model("models/best_improved_model.h5")

    if basic_model and improved_model:
        st.success("Both models loaded successfully!")

        # Model architecture comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Model")
            st.info("""
            **Features**:
            - Simple CNN architecture
            - 3 Conv2D layers (32, 64, 128 filters)
            - Basic training without augmentation
            - Dense layer: 256 neurons
            """)

        with col2:
            st.subheader("Improved Model")
            st.info("""
            **Features**:
            - Enhanced CNN architecture
            - 3 Conv2D layers (64, 128, 256 filters)
            - Data augmentation included
            - Dense layer: 512 neurons
            - Lower learning rate (0.0001)
            """)

        # Upload image for comparison
        st.subheader("Compare Predictions")
        uploaded_file = st.file_uploader(
            "Upload an image to compare model predictions",
            type=['png', 'jpg', 'jpeg'],
            key="comparison"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Display image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Image for Comparison",
                         use_column_width=True)

            # Get predictions from both models
            processed_image = preprocess_image(image)

            with st.spinner("Getting predictions from both models..."):
                basic_pred_class, basic_confidence, basic_probs = predict_image(
                    basic_model, processed_image)
                improved_pred_class, improved_confidence, improved_probs = predict_image(
                    improved_model, processed_image)

            if basic_pred_class is not None and improved_pred_class is not None:
                # Display comparison results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Basic Model Result")
                    basic_class_name = CLASS_NAMES[basic_pred_class] if basic_pred_class < len(
                        CLASS_NAMES) else f"Class_{basic_pred_class}"
                    st.info(f"**Prediction**: {basic_class_name}")
                    st.info(f"**Confidence**: {basic_confidence:.2%}")

                with col2:
                    st.subheader("Improved Model Result")
                    improved_class_name = CLASS_NAMES[improved_pred_class] if improved_pred_class < len(
                        CLASS_NAMES) else f"Class_{improved_pred_class}"
                    st.success(f"**Prediction**: {improved_class_name}")
                    st.success(f"**Confidence**: {improved_confidence:.2%}")

                # Create comparison chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Handle class labels for plotting
                basic_num_classes = len(basic_probs)
                improved_num_classes = len(improved_probs)

                # Generate class labels based on actual predictions
                basic_labels = get_class_labels(basic_num_classes)
                improved_labels = get_class_labels(improved_num_classes)

                # Basic model probabilities
                ax1.bar(basic_labels, basic_probs, color='lightblue')
                ax1.set_title('Basic Model Predictions')
                ax1.set_ylabel('Probability')
                ax1.tick_params(axis='x', rotation=45)

                # Improved model probabilities
                ax2.bar(improved_labels, improved_probs, color='lightgreen')
                ax2.set_title('Improved Model Predictions')
                ax2.set_ylabel('Probability')
                ax2.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

                # Agreement analysis
                if basic_pred_class == improved_pred_class:
                    st.success("✅ Both models agree on the prediction!")
                else:
                    st.warning("⚠️ Models disagree on the prediction.")

                confidence_diff = improved_confidence - basic_confidence
                if confidence_diff > 0:
                    st.info(
                        f"📈 Improved model is {confidence_diff:.2%} more confident")
                elif confidence_diff < 0:
                    st.info(
                        f"📉 Basic model is {abs(confidence_diff):.2%} more confident")
                else:
                    st.info("🔄 Both models have similar confidence levels")

    else:
        st.warning(
            "Please ensure both models are trained and saved before using this comparison feature.")

# Page 4: Technical Details
elif page == "Technical Details":
    st.header("🔧 Technical Details")

    # Training configuration
    st.subheader("Training Configuration")
    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.code("""
        Basic Model Training:
        - Optimizer: Adam
        - Learning Rate: Default (0.001)
        - Loss: Sparse Categorical Crossentropy
        - Epochs: 20
        - Batch Size: 32
        - Validation Split: 20%
        """, language="text")

    with config_col2:
        st.code("""
        Improved Model Training:
        - Optimizer: Adam
        - Learning Rate: 0.0001 (reduced)
        - Loss: Sparse Categorical Crossentropy
        - Epochs: 30 (with early stopping)
        - Batch Size: 32
        - Data Augmentation: Yes
        - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        """, language="text")

    # Key techniques
    st.subheader("Key Techniques Used")

    techniques = [
        ("Convolutional Layers", "Extract spatial features from images"),
        ("MaxPooling", "Reduce spatial dimensions and computational load"),
        ("Batch Normalization", "Stabilize training and improve convergence"),
        ("Dropout", "Prevent overfitting by randomly setting neurons to zero"),
        ("Data Augmentation", "Increase dataset diversity with transformations"),
        ("Early Stopping", "Prevent overtraining by monitoring validation loss"),
        ("Learning Rate Scheduling", "Reduce learning rate when loss plateaus"),
        ("Model Checkpointing", "Save best model based on validation accuracy")
    ]

    for technique, description in techniques:
        with st.expander(f"📋 {technique}"):
            st.write(description)

    # Performance metrics explanation
    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Accuracy**: Percentage of correctly classified images
        
        **Loss**: Measure of how far predictions are from true labels
        
        **Validation Accuracy**: Performance on unseen data during training
        """)

    with col2:
        st.info("""
        **Confusion Matrix**: Shows correct vs incorrect predictions per class
        
        **Classification Report**: Detailed precision, recall, and F1-score per class
        
        **Training Curves**: Visualize learning progress over epochs
        """)

    # Usage instructions
    st.subheader("How to Use This App")
    st.markdown("""
    1. **Model Overview**: Get familiar with the dataset and model architecture
    2. **Image Classification**: Upload images to see real-time predictions
    3. **Model Comparison**: Compare performance between basic and improved models
    4. **Technical Details**: Understand the techniques and configurations used
    
    **Tips**:
    - Upload clear images for better predictions
    - Try different image types (buildings, nature scenes, streets, etc.)
    - Compare results between models to see improvement
    - Check confidence scores to understand model certainty
    """)

# Footer
st.markdown("---")
st.markdown("**Built with**: TensorFlow/Keras, Streamlit, Python")
st.markdown("**Dataset**: Intel Image Classification Dataset")
