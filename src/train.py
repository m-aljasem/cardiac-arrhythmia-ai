"""
Training script for ECG classification with CNN-LSTM-Attention.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from model import ECGAttentionModel


class ECGDataset(Dataset):
    """ECG Dataset class."""
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def load_data(csv_path):
    """Load ECG data from CSV."""
    df = pd.read_csv(csv_path, header=None)
    signals = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return signals, labels


def train_model(csv_path='../data/data.csv', epochs=50, batch_size=32, lr=0.001):
    """Main training function."""
    print("Training ECG classification model...")
    
    # Load data
    signals, labels = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGAttentionModel(num_classes=5).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for signals_batch, labels_batch in train_loader:
            signals_batch, labels_batch = signals_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals_batch, labels_batch in test_loader:
            signals_batch, labels_batch = signals_batch.to(device), labels_batch.to(device)
            outputs = model(signals_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), '../models/ecg_model.pth')
    print("Model saved!")


if __name__ == '__main__':
    train_model()

