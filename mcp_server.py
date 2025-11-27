"""
MCP Server for ECG Arrhythmia Detection

Model Context Protocol (MCP) server that exposes the model as tools
for AI assistants and other MCP clients.
"""

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import io
import base64

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("⚠️  MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.model import ECGAttentionModel

app = Server("ecg-mcp-server")

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "ecg_model.pth"

# Global model
model = None
CLASSES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
NUM_CLASSES = len(CLASSES)
EXPECTED_FEATURES = (187,)

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
            print(f"⚠️  Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="predict",
            description="Make a prediction using the ECG Arrhythmia Detection model. Input should be a file path to a CSV file with ECG signal data ((187,) features) or base64 encoded CSV.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "File path to CSV file or base64 encoded CSV data"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="model_info",
            description="Get information about the loaded model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="health_check",
            description="Check if the model is loaded and ready",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    if name == "health_check":
        model_loaded = model is not None
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "healthy" if model_loaded else "degraded",
                "model_loaded": model_loaded
            }, indent=2)
        )]
    
    elif name == "model_info":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded"
                }, indent=2)
            )]
        
        info = {
            "model_type": "PyTorch",
            "model_path": str(MODEL_PATH),
            "classes": CLASSES,
            "expected_features": EXPECTED_FEATURES,
            "description": "ECG Arrhythmia Detection"
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    
    elif name == "predict":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded. Please train the model first."
                }, indent=2)
            )]
        
        try:
            input_data = arguments.get("input", "")
            
            # Handle file path or base64 encoded CSV
            if Path(input_data).exists():
                df = pd.read_csv(input_data)
            elif input_data.startswith("data:text/csv"):
                # Base64 encoded CSV
                header, encoded = input_data.split(",", 1)
                csv_data = base64.b64decode(encoded)
                df = pd.read_csv(io.BytesIO(csv_data))
            else:
                # Try as base64 string
                try:
                    csv_data = base64.b64decode(input_data)
                    df = pd.read_csv(io.BytesIO(csv_data))
                except:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Invalid input. Provide file path to CSV or base64 encoded CSV."
                        }, indent=2)
                    )]
            
            # Extract signal (exclude last column if it's label)
            signal = df.iloc[:, :-1].values if df.shape[1] > EXPECTED_FEATURES else df.values
            
            # Ensure correct shape
            if signal.shape[1] != EXPECTED_FEATURES:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Expected {EXPECTED_FEATURES} features, got {signal.shape[1]}"
                    }, indent=2)
                )]
            
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
            
            result = {
                "prediction": CLASSES[pred_class],
                "class_index": pred_class,
                "confidence": float(confidence),
                "probabilities": probabilities
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Prediction error: {str(e)}"
                }, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, indent=2)
        )]

async def main():
    """Main entry point."""
    # Load model
    load_model()
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
