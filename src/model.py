"""ECG classification model with attention mechanism"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGAttentionModel(nn.Module):
    """CNN-LSTM model with attention for ECG classification."""
    
    def __init__(self, num_classes=5):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = nn.MultiheadAttention(256, num_heads=8)
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        
        # Attention
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.fc(x)
        return x

