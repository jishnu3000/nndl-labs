import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Lab 2: Activation Functions and Neural Networks")

st.sidebar.title("Choose what to see:")
section = st.sidebar.radio(
    "Pick a section:",
    ["Activation Functions", "Neural Network", "Comparison"]
)

def step_function(x):
    result = []
    for value in x:
        if value >= 0:
            result.append(1)
        else:
            result.append(0)
    return result

def sigmoid_function(x):
    result = []
    for value in x:
        sigmoid_value = 1 / (1 + np.exp(-value))
        result.append(sigmoid_value)
    return result

def tanh_function(x):
    result = []
    for value in x:
        exp_pos = np.exp(value)
        exp_neg = np.exp(-value)
        tanh_value = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        result.append(tanh_value)
    return result

def relu_function(x):
    result = []
    for value in x:
        if value > 0:
            result.append(value)
        else:
            result.append(0)
    return result

if section == "Activation Functions":
    st.header("Activation Functions")
    st.write("Here are the 4 activation functions from my notebook:")
    
    x_min = st.slider("Min value", -10, 0, -5)
    x_max = st.slider("Max value", 0, 10, 5)
    
    x = np.linspace(x_min, x_max, 100)
    
    step_y = step_function(x)
    sigmoid_y = sigmoid_function(x)
    tanh_y = tanh_function(x)
    relu_y = relu_function(x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, step_y, label='Step Function', linewidth=2)
    ax.plot(x, sigmoid_y, label='Sigmoid Function', linewidth=2)
    ax.plot(x, tanh_y, label='Tanh Function', linewidth=2)
    ax.plot(x, relu_y, label='ReLU Function', linewidth=2)
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Output')
    ax.set_title('All Activation Functions')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Test a specific value:")
    test_value = st.number_input("Enter a number:", value=0.0)
    
    st.write(f"For input {test_value}:")
    st.write(f"- Step: {step_function([test_value])[0]}")
    st.write(f"- Sigmoid: {sigmoid_function([test_value])[0]:.4f}")
    st.write(f"- Tanh: {tanh_function([test_value])[0]:.4f}")
    st.write(f"- ReLU: {relu_function([test_value])[0]}")

elif section == "Neural Network":
    st.header("Neural Network Training")
    st.write("Train a neural network on heart disease data!")
    
    try:
        data = pd.read_csv('heart.csv')
        st.write("Data loaded successfully!")
        st.write(f"Dataset shape: {data.shape}")
        st.dataframe(data.head())
        
        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        
        X = data.drop('HeartDisease', axis=1).values
        y = data['HeartDisease'].values
        
        st.write("Data preprocessed!")
        
        st.subheader("Network Settings:")
        activation = st.selectbox("Choose activation:", ["relu", "tanh", "logistic"])
        hidden_size = st.slider("Hidden layer size:", 4, 16, 8)
        
        if st.button("Train Network"):
            st.write("Training...")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                activation=activation,
                max_iter=500,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            st.write("Training complete!")
            st.write(f"Training accuracy: {train_score:.3f}")
            st.write(f"Test accuracy: {test_score:.3f}")
            
            predictions = model.predict(X_test[:5])
            actual = y_test[:5]
            
            results_df = pd.DataFrame({
                'Actual': actual,
                'Predicted': predictions,
                'Correct?': ['Yes' if a == p else 'No' for a, p in zip(actual, predictions)]
            })
            
            st.write("Sample predictions:")
            st.dataframe(results_df)
    
    except FileNotFoundError:
        st.error("Heart.csv file not found! Please make sure the data folder exists.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

elif section == "Comparison":
    st.header("Compare Different Activations")
    st.write("Let's see which activation function works best!")
    
    try:
        data = pd.read_csv('heart.csv')
        
        categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        
        X = data.drop('HeartDisease', axis=1).values
        y = data['HeartDisease'].values
        
        if st.button("Compare All Activations"):
            st.write("Training 3 different networks...")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            activations = ['relu', 'tanh', 'logistic']
            results = []
            
            for activation in activations:
                st.write(f"Training {activation}...")
                
                model = MLPClassifier(
                    hidden_layer_sizes=(8,),
                    activation=activation,
                    max_iter=500,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                
                results.append({
                    'Activation': activation,
                    'Train Accuracy': f"{train_acc:.3f}",
                    'Test Accuracy': f"{test_acc:.3f}"
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            activations = [r['Activation'] for r in results]
            train_scores = [float(r['Train Accuracy']) for r in results]
            test_scores = [float(r['Test Accuracy']) for r in results]
            
            x_pos = np.arange(len(activations))
            width = 0.35
            
            ax.bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.8)
            ax.bar(x_pos + width/2, test_scores, width, label='Test', alpha=0.8)
            
            ax.set_xlabel('Activation Function')
            ax.set_ylabel('Accuracy')
            ax.set_title('Performance Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(activations)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            best_test = max(test_scores)
            best_activation = activations[test_scores.index(best_test)]
            st.write(f"Best activation: {best_activation} with test accuracy {best_test:.3f}")
    
    except FileNotFoundError:
        st.error("Heart.csv file not found!")
    except Exception as e:
        st.error(f"Error: {e}")