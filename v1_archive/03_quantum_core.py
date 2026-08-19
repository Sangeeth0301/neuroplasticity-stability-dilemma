import pennylane as qml
import numpy as np

# Phase 2: Quantum State Preparation & VQC Construction

# Task 2.1: Set up the 9-qubit environment
n_qubits = 9
dev = qml.device("default.qubit", wires=n_qubits)

# Define the number of layers for the VQC
n_layers = 3

# Initialize random parameters for the VQC
# Shape: (n_layers, n_qubits, 3) for StronglyEntanglingLayers
np.random.seed(42)
initial_weights = np.random.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits, 3))

@qml.qnode(dev, interface="autograd")
def vqc_circuit(weights, features):
    """
    The main 9-Qubit Variational Quantum Circuit (VQC).
    """
    # Task 2.2: Implement Quantum Amplitude Encoding
    # features must be a normalized 512-dimensional vector
    qml.AmplitudeEmbedding(features=features, wires=range(n_qubits), normalize=True)
    
    # Task 2.3: Construct the parameterized VQC (Hardware-efficient ansatz)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # We measure the expectation value of the PauliZ operator on the first qubit
    # as a simple output for testing the circuit construction.
    return qml.expval(qml.PauliZ(0))

def test_vqc_construction():
    print("Loading normalized quantum tensors...")
    try:
        tensors = np.load('quantum_ready_tensors.npy')
    except FileNotFoundError:
        print("Error: quantum_ready_tensors.npy not found. Did Phase 1 finish?")
        return
        
    print(f"Successfully loaded tensors with shape {tensors.shape}")
    
    # Take the first conversational turn to test the circuit
    test_feature = tensors[0]
    
    print(f"Testing VQC with 9 qubits and {n_layers} StronglyEntanglingLayers...")
    
    # Forward pass through the quantum circuit
    result = vqc_circuit(initial_weights, test_feature)
    
    print(f"Circuit successfully executed! Expectation value (PauliZ on qubit 0): {result:.4f}")
    
    # Draw the circuit for visualization (first 5 gates to avoid spam)
    print("\nCircuit Representation:")
    print(qml.draw(vqc_circuit, max_length=60)(initial_weights, test_feature))

if __name__ == "__main__":
    test_vqc_construction()
