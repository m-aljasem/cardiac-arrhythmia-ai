# ❤️ ECG Classification with CNN‑LSTM‑Attention

Deep neural network for classifying **ECG heartbeats** into 5 arrhythmia categories using 1D CNNs, Bi‑LSTMs, and an attention mechanism (PyTorch).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Classes](#-classes)
- [Exported Weights](#-exported-weights)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project implements a **sequence model** for ECG beat classification using:

- 1D CNN layers for local feature extraction  
- Bidirectional LSTM for temporal context  
- Multi‑head attention for focusing on critical time steps  

It uses the **MIT‑BIH ECG heartbeat dataset** and provides:

- A PyTorch training script
- A Streamlit UI for quick inference
- Exported `.pth` weights for deployment

---

## ✨ Features

- Robust CNN‑LSTM‑Attention architecture
- 5‑class heartbeat classification
- PyTorch training pipeline
- Streamlit app for CSV‑based ECG signals

---

## 🛠 Tech Stack

- Python 3.8+
- PyTorch
- NumPy / pandas
- Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
cd cardiac-arrhythmia-ai
python src/train.py
```

This will:

- Load ECG data from `data/` (MIT‑BIH heartbeat CSV files)
- Train `ECGAttentionModel`
- Save weights to:

```text
models/ecg_model.pth
```

### 2️⃣ Run the Streamlit App

```bash
cd cardiac-arrhythmia-ai
streamlit run app.py
```

Upload an ECG CSV (one or more beats), visualize the signal, and get predicted arrhythmia classes.

---

## 🧑‍💻 Usage

### 🌐 Web App

```bash
streamlit run app.py
```

The app:

- Builds `ECGAttentionModel`
- Loads `models/ecg_model.pth` if available
- Runs inference on uploaded ECG sequences

### 🧬 Programmatic Usage

```python
import torch
from src.model import ECGAttentionModel

model = ECGAttentionModel(num_classes=5)
state_dict = torch.load("models/ecg_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# signal: (batch, 1, sequence_length)
with torch.no_grad():
    logits = model(signal)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)
```

---

## 🗂 Project Structure

```text
cardiac-arrhythmia-ai/
├── app.py                    # Streamlit app
├── config/
├── data/                     # ECG CSV files (train/test)
├── docs/
├── experiments/
├── models/                   # Saved weights (ecg_model.pth)
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── model.py              # ECGAttentionModel
└── tests/
```

---

## 🧬 Classes

The model predicts 5 classes:

1. **Normal**  
2. **Artial Premature**  
3. **Premature Ventricular Contraction (PVC)**  
4. **Fusion of Ventricular and Normal**  
5. **Fusion of Paced and Normal**  

---

## 📦 Exported Weights

- Training script saves to:

```text
../models/ecg_model.pth
```

- Streamlit app loads from:

```text
models/ecg_model.pth
```

You can package `ecg_model.pth` with any other PyTorch‑based client for inference.

---

## 📄 License

Licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 🏥 Disclaimer

> This ECG classifier is for **research and educational purposes only**.  
> It must **not** be used for clinical diagnosis or patient monitoring.

