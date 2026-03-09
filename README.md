# Vision Transformer & Quantum Vision Transformer on MNIST

>  Classical ViT from scratch · Attention Rollout · Hybrid QViT · PennyLane · Cirq

---

## Overview

This project implements:

1. **Classical Vision Transformer (ViT)** on 10-class MNIST, built from scratch in TensorFlow/Keras following Dosovitskiy et al. (ICLR 2021) exactly — patch embedding, learnable CLS token, positional embeddings, multi-head self-attention, pre-LayerNorm transformer blocks, and cosine-decay training.

2. **Hybrid Quantum Vision Transformer (QViT)** extending the classical ViT by replacing the MLP sub-block with a **Parameterised Quantum Circuit (PQC)** and implementing quantum attention via the **SWAP test**. Built with PennyLane (Cirq backend) + TensorFlow/Keras.

3. **Research paper** (LaTeX/PDF) covering the full theoretical background, architecture design, complexity analysis, and implementation challenges.

---

## Files

```
├── vit_mnist_pennylane.ipynb   # Main notebook (classical ViT + hybrid QViT)
├── vit_paper.pdf               # 8-page IEEE-style research paper
├── vit_paper.tex               # LaTeX source
└── README.md                   # This file
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
pip install pennylane pennylane-cirq cirq
```

> **Note:** `pennylane-cirq` is optional. If not installed, the notebook automatically falls back to `pennylane`'s built-in `default.qubit` simulator.

### 2. Run the notebook

Open `vit_mnist_pennylane.ipynb` in Jupyter or Google Colab and run cells sequentially.

```bash
jupyter notebook vit_mnist_pennylane.ipynb
```

The notebook is self-contained — MNIST is downloaded automatically via `keras.datasets`.

---

## Classical ViT Architecture

Follows the ViT paper (Dosovitskiy et al., 2021) with the following configuration:

| Hyperparameter | Value |
|---|---|
| Image size | 28 × 28 |
| Patch size | 4 × 4 |
| Number of patches | 49 |
| Embedding dim | 64 |
| Transformer layers | 6 |
| Attention heads | 4 |
| MLP hidden dim | 128 |
| Dropout | 0.1 |
| Optimiser | AdamW + warmup cosine decay |

**Key implementation details:**
- Pre-LayerNorm formulation (more stable than post-LN)
- GELU activations in MLP blocks
- Learnable 1D positional embeddings
- Warmup cosine decay learning rate schedule
- Attention rollout visualisation (Abnar & Zuidema, 2020)

---

## Hybrid QViT Architecture

The QViT replaces the classical MLP block with a **data re-uploading PQC** (Pérez-Salinas et al., 2020):

```
Classical ViT          →   Hybrid QViT
─────────────────────────────────────────────────────
Linear Patch Proj.     →   Angle encoding  qml.RY(x·π)
Multi-Head Attention   →   SWAP test  |⟨ψQ|ψK⟩|²
MLP Block              →   PQC: RY(θ) · RZ(φ) + CNOT ring
Positional Embedding   →   Quantum phase encoding  RZ(2πki/N)
CLS Token              →   Ancilla qubit  |0⟩
Gradient               →   Parameter-shift rule
```

**QViT configuration (binary MNIST, 0 vs 1):**

| Hyperparameter | Value |
|---|---|
| Image size | 8 × 8 |
| Patch size | 4 × 4 |
| Qubits per token | 4 |
| PQC depth | 2 layers |
| Variational parameters | 16 |
| Encoding | Data re-uploading |
| Gradient method | Parameter-shift rule |
| Backend | PennyLane + Cirq |

### Quantum Attention (SWAP Test)

Attention score between patch tokens $i$ and $j$:

$$A_{ij} = |\langle\psi_Q|\psi_K\rangle|^2 = 2\,P(\text{ancilla}=|0\rangle) - 1$$

Implemented in Cirq using `cirq.CSWAP` gates on a `2n+1` qubit register.

### PQC Circuit (per token, per encoder block)

```
q0: ──RY(x·π)──RY(θ)──RZ(φ)──●──────────────X──  → ⟨Z₀⟩
q1: ──RY(x·π)──RY(θ)──RZ(φ)──X──●───────────|───  → ⟨Z₁⟩
q2: ──RY(x·π)──RY(θ)──RZ(φ)──────X──●───────|───  → ⟨Z₂⟩
q3: ──RY(x·π)──RY(θ)──RZ(φ)─────────X───────●───  → ⟨Z₃⟩
     [encode]  [variational]  [CNOT ring]
```

---

## Notebook Contents

| Part | Description |
|---|---|
| 1 | Setup & imports |
| 2 | MNIST data loading & visualisation |
| 3 | ViT hyperparameters |
| 4 | ViT building blocks (PatchEmbedding, CLSToken, MHSA, EncoderBlock) |
| 5 | Training with warmup cosine decay |
| 6 | Training curves |
| 7 | Test evaluation + per-class report |
| 8 | Attention rollout visualisation |
| 9 | Positional embedding cosine similarity |
| 10 | Misclassification analysis |
| 11 | Hybrid QViT: PennyLane + Cirq setup |
| 11.3 | Binary MNIST data preparation |
| 11.4 | PennyLane QNode + circuit diagram |
| 11.5 | SWAP test quantum attention (Cirq) |
| 11.6 | Hybrid QViT Keras model |
| 11.7 | Training |
| 11.8 | Test evaluation + confusion matrix |
| 11.9 | Training curves |
| 11.10 | Learned PQC weight visualisation |
| 11.11 | Classical ViT vs QViT comparison table |
| 11.12 | Architecture diagram |
| 12 | Complexity comparison table |
| 13 | References |

---

## Known Issues & Fixes

Several compatibility issues arise between PennyLane, Keras 3, and GPU environments. All are resolved in the notebook:

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: qml.qnn has no attribute 'KerasLayer'` | Removed in newer PennyLane | Custom `PennyLaneLayer` wrapping the QNode |
| `SymbolicTensor cannot be interpreted as int` | Keras traces `call()` symbolically | Use `tf.map_fn` or `tf.py_function` |
| `EagerPyFunc` XLA error | GPU JIT can't compile Python callbacks | `run_eagerly=True` + `set_jit(False)` |
| `float64/float32 Mul mismatch` | PennyLane uses float64 internally | Cast inputs to `tf.float64`, outputs back to `tf.float32` |

---

## Performance

**Classical ViT (10-class MNIST):**
- Test accuracy: ≥ 99.0%
- Parameters: ~200K
- Training time: ~5 min (GPU)

**Hybrid QViT (binary MNIST, 0 vs 1):**
- Expected test accuracy: > 95%
- Parameters: 16 quantum + ~200 classical
- Training time: ~30s/epoch with optimisations (see below)

### Speed Tips for QViT

By default, quantum simulation is slow (~400s/epoch). Apply these to get ~30s/epoch:

```python
# 1. Faster simulator (C++ accelerated)
dev = qml.device('lightning.qubit', wires=N_QUBITS)

# 2. Faster gradient method (adjoint on simulators)
@qml.qnode(dev, interface='tf', diff_method='best')

# 3. Reduce dataset
x_qtrain_feat = x_qtrain_feat[:500]

# 4. Small batches
Q_BATCH = 8

# 5. Shallower circuit
N_PQC_LAYERS = 1
```

---

## References

1. Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
3. Bergholm et al. (2018). *PennyLane.* [arXiv:1811.04968](https://arxiv.org/abs/1811.04968)
4. Cherrat et al. (2022). *Quantum Vision Transformers.* [arXiv:2209.08167](https://arxiv.org/abs/2209.08167)
5. Pérez-Salinas et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum 4, 226.
6. Mitarai et al. (2018). *Quantum circuit learning.* Phys. Rev. A 98, 032309. *(parameter-shift rule)*
7. Buhrman et al. (2001). *Quantum Fingerprinting.* PRL 87, 167902. *(SWAP test)*
8. Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL 2020.
9. McClean et al. (2018). *Barren plateaus in quantum neural network training.* Nature Comms 9, 4812.
10. Mari et al. (2020). *Transfer learning in hybrid classical-quantum neural networks.* npj Quantum Inf 6.
