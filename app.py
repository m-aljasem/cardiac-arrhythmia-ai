"""Streamlit app for ECG classification"""
import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.explainability import ModelExplainer
from src.model import ECGAttentionModel

LABELS = ["Normal", "Artial Premature", "Premature ventricular contraction",
          "Fusion of ventricular and normal", "Fusion of paced and normal"]

st.set_page_config(page_title="Cardiac Arrhythmia AI", page_icon="❤️")
st.title("Cardiac Arrhythmia AI")

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "ecg_model.pth"

uploaded_file = st.file_uploader("Upload ECG signal (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    signal = df.iloc[:, :-1].values  # Exclude last column (label)
    
    st.line_chart(signal[0] if len(signal) > 0 else signal)
    
    if st.button("Classify"):
        # Prepare input
        signal_tensor = torch.FloatTensor(signal).unsqueeze(1)  # Add channel dim
        
        # Load model
        model = ECGAttentionModel(num_classes=5)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if WEIGHTS_PATH.exists():
            st.info(f"Loading trained weights from `{WEIGHTS_PATH}`")
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        else:
            st.warning(
                "⚠️ Trained weights not found. Using randomly initialized model "
                "(predictions will be unreliable). Train the model first."
            )
        model.eval()
        
        with torch.no_grad():
            pred = model(signal_tensor)
            pred_class = torch.argmax(pred, dim=1)[0].item()
            confidence = torch.softmax(pred, dim=1)[pred_class].item()
        
        st.success(f"**Prediction**: {LABELS[pred_class]} (Confidence: {confidence:.2%})")
        
        # Explainability
        st.divider()
        st.subheader("🔍 Explainability")
        st.info("SHAP explanations for PyTorch sequence models require additional setup. See docs/EXPLAINABILITY.md for details.")

