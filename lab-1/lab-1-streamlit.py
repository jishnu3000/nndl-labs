import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

st.title("Lab 1: Logic Gates with Perceptron")

st.sidebar.title("Choose a logic gate:")
gate_type = st.sidebar.radio(
    "Pick a gate:",
    ["AND", "OR", "AND-NOT", "XOR"]
)

def create_gate_dataset(gate_type):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    if gate_type == 'AND':
        y = np.array([0, 0, 0, 1])
    elif gate_type == 'OR':
        y = np.array([0, 1, 1, 1])
    elif gate_type == 'AND-NOT':
        y = np.array([0, 0, 1, 0])
    elif gate_type == 'XOR':
        y = np.array([0, 1, 1, 0])
    
    return X, y

X, y = create_gate_dataset(gate_type)

st.header(f"{gate_type} Gate Truth Table")
truth_table = pd.DataFrame({
    'Input 1': X[:, 0],
    'Input 2': X[:, 1],
    'Output': y
})
st.dataframe(truth_table)

st.header("Training Settings")
learning_rate = st.slider("Learning Rate", 0.1, 2.0, 1.0, 0.1)
max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

if st.button(f"Train {gate_type} Gate"):
    st.write("Training perceptron...")
    
    perceptron = Perceptron(
        eta0=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    
    perceptron.fit(X, y)
    
    predictions = perceptron.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    st.write("Training completed!")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Iterations used: {perceptron.n_iter_}")
    
    results_df = pd.DataFrame({
        'Input 1': X[:, 0],
        'Input 2': X[:, 1],
        'Expected': y,
        'Predicted': predictions,
        'Correct?': ['Yes' if pred == exp else 'No' for pred, exp in zip(predictions, y)]
    })
    
    st.write("Results:")
    st.dataframe(results_df)
    
    st.write("Learned Parameters:")
    st.write(f"Weight 1: {perceptron.coef_[0][0]:.3f}")
    st.write(f"Weight 2: {perceptron.coef_[0][1]:.3f}")
    st.write(f"Bias: {perceptron.intercept_[0]:.3f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=150, edgecolor='black')
    
    for i, (x_coord, y_coord) in enumerate(X):
        ax.text(x_coord + 0.05, y_coord + 0.05, f'({x_coord},{y_coord})\nOut: {y[i]}', 
                fontsize=10)
    
    if perceptron.coef_[0][1] != 0:
        w1, w2 = perceptron.coef_[0]
        b = perceptron.intercept_[0]
        x_line = np.linspace(-0.5, 1.5, 100)
        y_line = -(w1 * x_line + b) / w2
        ax.plot(x_line, y_line, 'g--', linewidth=2, label='Decision Boundary')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(f'{gate_type} Gate Decision Boundary')
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)
    
    if accuracy == 1.0:
        st.success(f"Success! The perceptron learned the {gate_type} gate perfectly!")
    else:
        st.error(f"Failed! The perceptron could not learn the {gate_type} gate perfectly.")
        if gate_type == "XOR":
            st.write("This is expected because XOR is not linearly separable!")

if gate_type == "XOR":
    st.header("About XOR Gate")
    st.write("The XOR gate is special - it cannot be learned by a single perceptron!")
    st.write("XOR is 'not linearly separable' which means no straight line can separate the classes.")
    
    if st.button("Try Multi-Layer Perceptron for XOR"):
        from sklearn.neural_network import MLPClassifier
        
        st.write("Using MLPClassifier (Multi-Layer Perceptron)...")
        
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
        mlp.fit(X, y)
        
        mlp_predictions = mlp.predict(X)
        mlp_accuracy = accuracy_score(y, mlp_predictions)
        
        st.write(f"MLP Accuracy: {mlp_accuracy:.2%}")
        
        if mlp_accuracy == 1.0:
            st.success("Success! The Multi-Layer Perceptron solved XOR!")
        
        mlp_results = pd.DataFrame({
            'Input 1': X[:, 0],
            'Input 2': X[:, 1],
            'Expected': y,
            'MLP Predicted': mlp_predictions,
            'Correct?': ['Yes' if pred == exp else 'No' for pred, exp in zip(mlp_predictions, y)]
        })
        
        st.dataframe(mlp_results)

st.header("Gate Comparison")
if st.button("Compare All Gates"):
    st.write("Testing all gates with perceptron...")
    
    gates = ['AND', 'OR', 'AND-NOT', 'XOR']
    results = []
    
    for gate in gates:
        X_test, y_test = create_gate_dataset(gate)
        
        perceptron_test = Perceptron(max_iter=1000, random_state=42)
        perceptron_test.fit(X_test, y_test)
        
        accuracy = perceptron_test.score(X_test, y_test)
        
        results.append({
            'Gate': gate,
            'Accuracy': f"{accuracy:.1%}",
            'Can Learn?': 'Yes' if accuracy == 1.0 else 'No'
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    gate_names = [r['Gate'] for r in results]
    accuracies = [float(r['Accuracy'].replace('%', ''))/100 for r in results]
    colors = ['green' if acc == 1.0 else 'red' for acc in accuracies]
    
    bars = ax.bar(gate_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Perceptron Performance on Different Logic Gates')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                f'{acc:.1%}', ha='center', fontweight='bold')
    
    st.pyplot(fig)