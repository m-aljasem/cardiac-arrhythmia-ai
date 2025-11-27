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




## 🌐 RESTful API

The project includes a FastAPI server for programmatic access to the model. This allows you to integrate predictions into your own applications, web services, or scripts.

### Installation

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Starting the API Server

Start the API server using one of these methods:

**Method 1: Direct Python execution**
```bash
python api.py
```

**Method 2: Using uvicorn directly**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Production mode (no auto-reload)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check endpoint (checks if model is loaded) |
| `/model/info` | GET | Get detailed model information |
| `/predict` | POST | Make a prediction |

### Interactive API Documentation

Once the server is running, you can access interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with "Try it out" feature
- **ReDoc**: `http://localhost:8000/redoc` - Beautiful, responsive API documentation

### Using the API

#### Health Check

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
# Output: {"status": "healthy", "model_loaded": true}
```

#### Get Model Information

```python
import requests

response = requests.get("http://localhost:8000/model/info")
print(response.json())
# Output: Model type, input shape, classes, etc.
```

#### Make Predictions

# Example: ECG Arrhythmia Detection
import requests
import pandas as pd
import numpy as np

# Prepare ECG signal data (187 features)
# Your ECG signal should be a 1D array of 187 values
ecg_signal = np.random.randn(187)  # Replace with actual ECG data

# Create DataFrame
df = pd.DataFrame([ecg_signal])

# Save to CSV
df.to_csv("ecg_signal.csv", index=False, header=False)

# Upload CSV file
with open("ecg_signal.csv", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()

print(f"Arrhythmia Type: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['probabilities']}")

### Using cURL

You can also use cURL to interact with the API:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model/info
```

**Make Prediction (for image-based models):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

**Make Prediction (for JSON-based models like Alzheimer's):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": 1.0, "age": 75.0, "education": 14.0, "ses": 2.0, "mmse": 27.0, "etiv": 1490.0, "nwbv": 0.73, "asf": 1.20}'
```

### Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid input data
- `503 Service Unavailable` - Model not loaded (train the model first)
- `500 Internal Server Error` - Server error during prediction

Example error handling:

```python
import requests

try:
    response = requests.post("http://localhost:8000/predict", json=data)
    response.raise_for_status()  # Raises exception for bad status codes
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

### API Response Format

**Successful Prediction Response:**
```json
{
    "prediction": "Demented",
    "confidence": 0.85,
    "probabilities": {
        "Nondemented": 0.15,
        "Demented": 0.85
    }
}
```

**Error Response:**
```json
{
    "detail": "Model not loaded. Please train the model first."
}
```

### Deployment

For production deployment, consider:

1. **Using a production ASGI server**: Use `uvicorn` with multiple workers or `gunicorn` with uvicorn workers
2. **Adding authentication**: Implement API keys or OAuth2
3. **Rate limiting**: Add rate limiting to prevent abuse
4. **Logging**: Configure proper logging for monitoring
5. **HTTPS**: Use HTTPS in production with SSL certificates

Example production command:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server that exposes the model as tools for AI assistants and other MCP-compatible clients. This allows AI assistants like Claude, ChatGPT, or custom MCP clients to interact with your model.

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol for AI assistants to interact with external tools and services. It enables AI assistants to:
- Call your model for predictions
- Get model information
- Check model health status

### Installation

The MCP server requires the MCP SDK:

```bash
pip install mcp
```

### Starting the MCP Server

Start the MCP server:

```bash
python mcp_server.py
```

The server runs as a stdio-based server, communicating via standard input/output. It's designed to be used with MCP clients.

### MCP Tools

The server exposes the following tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict` | Make a prediction using the model | `input` (string): Input data as JSON string or file path |
| `model_info` | Get information about the loaded model | None |
| `health_check` | Check if the model is loaded and ready | None |

### Using MCP with Python Client

```python
from mcp import ClientSession, StdioServerParameters
import asyncio
import json

async def main():
    # Connect to MCP server
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )
    ) as session:
        # Initialize the session
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        
        # Health check
        health_result = await session.call_tool(
            "health_check",
            {}
        )
        print("Health:", health_result.content[0].text)
        
        # Get model info
        model_info = await session.call_tool(
            "model_info",
            {}
        )
        print("Model Info:", model_info.content[0].text)
        
        # Make prediction
        # For image-based models, provide base64 encoded image or file path
        prediction_input = json.dumps({
            "file_path": "test_image.jpg"
        })
        
        prediction_result = await session.call_tool(
            "predict",
            {"input": prediction_input}
        )
        print("Prediction:", prediction_result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP with Claude Desktop

To use with Claude Desktop, add this to your MCP configuration file:

```json
{
  "mcpServers": {
    "ecg": {
      "command": "python",
      "args": ["/home/m-aljasem/projects/ai-projects/ecg/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/home/m-aljasem/projects/ai-projects/ecg"
      }
    }
  }
}
```

### MCP Tool Responses

**Health Check Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Model Info Response:**
```json
{
  "model_type": "TensorFlow/Keras",
  "model_path": "models/model.h5",
  "classes": ["Class1", "Class2"],
  "description": "Model description"
}
```

**Prediction Response:**
```json
{
  "prediction": "Class1",
  "confidence": 0.95,
  "probabilities": {
    "Class1": 0.95,
    "Class2": 0.05
  }
}
```

### Error Handling

The MCP server returns error messages in JSON format:

```json
{
  "error": "Model not loaded. Please train the model first."
}
```

### Integration Examples

**Example 1: Batch Predictions**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters

async def batch_predict(file_paths):
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        results = []
        for file_path in file_paths:
            result = await session.call_tool(
                "predict",
                {"input": json.dumps({"file_path": file_path})}
            )
            results.append(json.loads(result.content[0].text))
        
        return results

# Usage
predictions = asyncio.run(batch_predict([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]))
```

**Example 2: Model Monitoring**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
import time

async def monitor_model():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        while True:
            health = await session.call_tool("health_check", {})
            print(f"[{time.strftime('%H:%M:%S')}] Health: {health.content[0].text}")
            await asyncio.sleep(60)  # Check every minute

# Run monitoring
asyncio.run(monitor_model())
```

### Troubleshooting

**Issue: MCP server not starting**
- Ensure `mcp` package is installed: `pip install mcp`
- Check that the model file exists in the `models/` directory
- Verify Python path and dependencies

**Issue: Tool calls failing**
- Check that the model is trained and weights are saved
- Verify input format matches expected format
- Check server logs for detailed error messages

**Issue: Connection errors**
- Ensure the MCP server process is running
- Check that stdio communication is working
- Verify environment variables if needed

