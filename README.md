# Dual-Manifold ContextSync (V2)

## Stability-Plasticity in Quantum-Classical Hybrid LLMs

This repository contains the architecture and codebase for the **Dual-Manifold ContextSync Architecture**, a hybrid quantum-classical system designed to solve the Stability-Plasticity dilemma in Large Language Models (LLMs).

### 🚀 Getting Started

**IMPORTANT:** Do NOT use any legacy V1 architecture files or references. The complete master blueprint for this project is located in:
👉 `Q_ContextSync_Build_Atlas.html`

Your friend should open this HTML file in their browser and read it from top to bottom. It contains the exact mathematical thesis, the 10-qubit Quantum Relevance Engine design, the classical Append-Only Stability Manifold, and the precise 10-phase execution plan.

### 📦 Prerequisites & Setup

A `requirements.txt` has been generated for you with all the necessary classical and quantum libraries (PennyLane, Qiskit, TensorCircuit, PyTorch, etc.). 

To install everything in your environment:
```bash
pip install -r requirements.txt
```

### 🗺️ Project Phases

As detailed in the Build Atlas, the project must be executed in strict order:
1. **Phase 00:** Foundation (Environment & CI)
2. **Phase 01:** Ground-Truth Dataset Assembly
3. **Phase 02:** Classical Stability Manifold (SQLite + FAISS)
4. **Phase 03:** The 10-Qubit VQC Relevance Engine (PennyLane)
5. **Phase 04-06:** Model Training & QFIM Anchoring
6. **Phase 07-10:** Evaluation & Integration

Good luck building!
