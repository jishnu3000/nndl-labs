import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Set page config
st.set_page_config(
    page_title="CIFAR-10 Neural Network Demo",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("CIFAR-10 Neural Network Classification Demo")
st.write("A beginner-friendly demonstration of image classification using neural networks")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose a section:", [
    "1. Dataset Overview",
    "2. Neural Network Architecture",
    "3. Model Training Demo",
    "4. Make Predictions"
])

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Helper function to load sample data


@st.cache_data
def load_sample_data():
    """Load a small sample of CIFAR-10 data for demo"""
    # Create some sample data (since we might not have the full dataset)
    np.random.seed(42)
    sample_images = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    sample_labels = np.random.randint(0, 10, 20)
    return sample_images, sample_labels

# Helper function to create a simple model


def create_simple_model():
    """Create a simplified neural network model"""
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(128, activation='relu', name='hidden_layer_1'),
        Dropout(0.3),
        Dense(64, activation='relu', name='hidden_layer_2'),
        Dropout(0.4),
        Dense(10, activation='softmax', name='output_layer')
    ])
    return model


# Section 1: Dataset Overview
if section == "1. Dataset Overview":
    st.header("📊 CIFAR-10 Dataset Overview")

    st.subheader("What is CIFAR-10?")
    st.write("""
    CIFAR-10 is a popular dataset for image classification containing:
    
    - **60,000 images** total
    - **50,000 training images**
    - **10,000 test images**
    - **32x32 pixel** color images
    - **10 different classes**
    """)

    # Show class names
    st.subheader("10 Classes in CIFAR-10:")
    for i, name in enumerate(class_names):
        st.write(f"{i}: {name.capitalize()}")

# Section 2: Neural Network Architecture
elif section == "2. Neural Network Architecture":
    st.header("🏗️ Neural Network Architecture")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Our Network Design")
        st.write("""
        **Input Layer:**
        - Takes 32×32×3 images (3072 pixels)
        - Flattens 2D images to 1D
        
        **Hidden Layers:**
        - Layer 1: 128 neurons + ReLU + Dropout
        - Layer 2: 64 neurons + ReLU + Dropout
        
        **Output Layer:**
        - 10 neurons (one for each class)
        - Softmax activation for probabilities
        """)

        # Interactive layer size adjustment
        st.subheader("Customize Architecture")
        layer1_size = st.slider("Hidden Layer 1 Size", 32, 512, 128, 32)
        layer2_size = st.slider("Hidden Layer 2 Size", 16, 256, 64, 16)
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.7, 0.3, 0.1)

        # Create and show model
        custom_model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(layer1_size, activation='relu'),
            Dropout(dropout_rate),
            Dense(layer2_size, activation='relu'),
            Dropout(dropout_rate),
            Dense(10, activation='softmax')
        ])

        st.subheader("Model Summary")
        # Create a simple summary
        total_params = (3072 * layer1_size + layer1_size +
                        layer1_size * layer2_size + layer2_size +
                        layer2_size * 10 + 10)

        st.write(f"**Total Parameters:** {total_params:,}")
        st.write(f"**Network Flow:**")
        st.write(
            f"Input (3072) → Dense({layer1_size}) → Dense({layer2_size}) → Output(10)")

    with col2:
        st.subheader("Why This Architecture?")

        with st.expander("🔍 Layer Sizes"):
            st.write("""
            - **Decreasing sizes** (128 → 64 → 10) help the network learn hierarchical features
            - **128 neurons** capture initial patterns from 3072 input features
            - **64 neurons** learn higher-level combinations
            - **10 neurons** map to final classes
            """)

        with st.expander("🔍 Activation Functions"):
            st.write("""
            **ReLU (Hidden Layers):**
            - Fast computation: f(x) = max(0, x)
            - Prevents vanishing gradients
            - Creates sparse activations
            
            **Softmax (Output):**
            - Converts scores to probabilities
            - All outputs sum to 1.0
            - Perfect for multi-class classification
            """)

        with st.expander("🔍 Dropout"):
            st.write("""
            - **Prevents overfitting** by randomly setting neurons to 0
            - Forces network to not rely on specific neurons
            - Improves generalization to new data
            - Only active during training
            """)

# Section 3: Model Training Demo
elif section == "3. Model Training Demo":
    st.header("📈 Model Training Demonstration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Process")
        st.write("""
        **Steps in Training:**
        1. **Forward Pass**: Data flows through network
        2. **Loss Calculation**: Compare predictions to true labels
        3. **Backpropagation**: Calculate gradients
        4. **Weight Update**: Adjust weights using optimizer
        5. **Repeat** for many epochs
        """)

        # Training parameters
        st.subheader("Training Settings")
        epochs = st.slider("Number of Epochs", 1, 50, 10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox(
            "Learning Rate", [0.0001, 0.001, 0.01, 0.1], index=1)

        st.write(f"**Selected Settings:**")
        st.write(f"- Epochs: {epochs}")
        st.write(f"- Batch Size: {batch_size}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Optimizer: Adam")
        st.write(f"- Loss Function: Categorical Crossentropy")

    with col2:
        st.subheader("Simulated Training Progress")

        if st.button("Start Training Simulation"):
            # Simulate training progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create simulated training data
            epochs_list = list(range(1, epochs + 1))

            # Simulate decreasing loss and increasing accuracy
            train_loss = [
                2.3 * np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.05) for i in epochs_list]
            train_acc = [0.1 + 0.8 * (1 - np.exp(-0.1 * i)) +
                         np.random.normal(0, 0.02) for i in epochs_list]
            val_loss = [2.5 * np.exp(-0.08 * i) + 0.2 +
                        np.random.normal(0, 0.08) for i in epochs_list]
            val_acc = [0.1 + 0.75 * (1 - np.exp(-0.08 * i)) +
                       np.random.normal(0, 0.03) for i in epochs_list]

            # Simulate training
            for i in range(epochs):
                progress_bar.progress((i + 1) / epochs)
                status_text.text(
                    f'Epoch {i+1}/{epochs} - Loss: {train_loss[i]:.4f} - Accuracy: {train_acc[i]:.4f}')

            # Plot training curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Loss plot
            ax1.plot(epochs_list, train_loss,
                     label='Training Loss', color='blue')
            ax1.plot(epochs_list, val_loss,
                     label='Validation Loss', color='red')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Accuracy plot
            ax2.plot(epochs_list, train_acc,
                     label='Training Accuracy', color='blue')
            ax2.plot(epochs_list, val_acc,
                     label='Validation Accuracy', color='red')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            status_text.text(
                f'Training Complete! Final Accuracy: {train_acc[-1]:.2%}')

# Section 4: Make Predictions
elif section == "4. Make Predictions":
    st.header("🔮 Make Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("How Prediction Works")
        st.write("""
        **Prediction Process:**
        1. **Input Image**: 32×32×3 pixel image
        2. **Preprocessing**: Normalize pixel values (0-255 → 0-1)
        3. **Forward Pass**: Image flows through trained network
        4. **Output Probabilities**: Get probability for each class
        5. **Final Prediction**: Class with highest probability
        """)

        # Simulate prediction
        st.subheader("Prediction Demo")

        if st.button("Generate Random Prediction"):
            # Create a random sample image
            sample_image = np.random.randint(
                0, 256, (32, 32, 3), dtype=np.uint8)

            # Simulate prediction probabilities
            np.random.seed()
            probabilities = np.random.dirichlet(
                np.ones(10) * 2)  # More realistic distribution
            predicted_class = np.argmax(probabilities)

            # Display image
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(sample_image)
            ax.set_title(f'Sample Image')
            ax.axis('off')
            st.pyplot(fig)

            st.write(f"**Predicted Class:** {class_names[predicted_class]}")
            st.write(f"**Confidence:** {probabilities[predicted_class]:.2%}")

    with col2:
        st.subheader("Class Probabilities")

        if 'probabilities' in locals():
            # Create DataFrame for probabilities
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)

            # Display as bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(range(len(class_names)), prob_df['Probability'],
                          color=['red' if i == predicted_class else 'lightblue'
                                 for i in range(len(class_names))])
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities for All Classes')
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(prob_df['Class'], rotation=45)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Show top 3 predictions
            st.subheader("Top 3 Predictions:")
            for i in range(3):
                class_name = prob_df.iloc[i]['Class']
                prob = prob_df.iloc[i]['Probability']
                st.write(f"{i+1}. **{class_name}**: {prob:.2%}")

# Footer
st.markdown("---")
st.markdown("""
**Key Learning Points:**
- Neural networks learn patterns in data through training
- Architecture design affects model performance
- Training involves iterative improvement through backpropagation
- Final models can classify new images with confidence scores

*This demo simplifies complex concepts for educational purposes.*
""")
