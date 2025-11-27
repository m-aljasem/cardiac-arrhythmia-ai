"""
RESTful API Server for ECG Arrhythmia Detection

This FastAPI server provides endpoints for model predictions and health checks.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.model import ECGAttentionModel

app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="RESTful API for ECG Arrhythmia Detection predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "ecg_model.pth"

# Global model
model = None
CLASSES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
NUM_CLASSES = len(CLASSES)
INPUT_SHAPE = (187,)
EXPECTED_FEATURES = 187

def load_model():
    """Load PyTorch model."""
    global model
    try:
        if MODEL_PATH.exists():
            model = ECGAttentionModel(num_classes=NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}. API will return errors until model is trained.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ECG Arrhythmia Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make a prediction from uploaded CSV file."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Extract signal (exclude last column if it's label)
        signal = df.iloc[:, :-1].values if df.shape[1] > EXPECTED_FEATURES else df.values
        
        # Ensure correct shape
        if signal.shape[1] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_FEATURES} features, got {signal.shape[1]}"
            )
        
        # Convert to tensor
        signal_tensor = torch.FloatTensor(signal).unsqueeze(1)  # Add channel dim
        
        # Predict
        with torch.no_grad():
            pred = model(signal_tensor)
            pred_class = torch.argmax(pred, dim=1)[0].item()
            confidence = torch.softmax(pred, dim=1)[pred_class].item()
        
        probabilities = {
            CLASSES[i]: float(torch.softmax(pred, dim=1)[0][i].item())
            for i in range(NUM_CLASSES)
        }
        
        return {
            "prediction": CLASSES[pred_class],
            "class_index": pred_class,
            "confidence": float(confidence),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "PyTorch",
        "input_shape": INPUT_SHAPE,
        "classes": CLASSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
