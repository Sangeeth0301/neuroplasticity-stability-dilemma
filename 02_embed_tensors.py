import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def embed_and_normalize():
    print("Loading HuggingFace Sentence Transformer (512-Dimensions)...")
    # This specific model outputs exactly 512 dimensions, perfect for 2^9 qubits
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    
    conn = sqlite3.connect('q_contextsync.db')
    df = pd.read_sql_query("SELECT * FROM chat_log ORDER BY turn_id", conn)
    conn.close()
    
    print("Embedding text into continuous semantic vectors...")
    raw_embeddings = model.encode(df['raw_text'].tolist())
    
    # L2-Normalization to prepare for Quantum Amplitude Encoding
    print("Applying L2-Normalization to fit the quantum state...")
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    normalized_embeddings = raw_embeddings / norms
    
    # Save the tensors for PennyLane to load directly
    np.save('quantum_ready_tensors.npy', normalized_embeddings)
    
    print(f"Success! Saved tensor array with shape: {normalized_embeddings.shape}")
    print("The dataset is now mathematically ready to be loaded into 9 qubits.")

if __name__ == "__main__":
    embed_and_normalize()