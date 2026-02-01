import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import io
import csv

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Local Modules
from gruModel import ImprovedGRUModel
from preprocessing import preprocess

class ImprovedGRUModel(nn.Module):
    def __init__(self, inputSize = 3, hiddenSize = 16, numLayers=2, dropout=0.2):
        super().__init__()
        
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        
        # GRU with dropout
        self.gru = nn.GRU(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0
        )
        
        # Layer normalization (replacion batch norm)
        self.ln = nn.LayerNorm(hiddenSize)
        
        # Hidden layer 1
        self.fc1 = nn.Linear(hiddenSize, hiddenSize // 2)
        self.dropout1 = nn.Dropout(dropout)

        # Hidden layer 2
        self.fc2 = nn.Linear(hiddenSize // 2, hiddenSize // 4)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.output = nn.Linear(hiddenSize // 4, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, return_attention=False):

        # GRU forward
        gruOut, _ = self.gru(x) 
        
        # Temporal pooling
        context = gruOut.mean(dim=1)

        # Normalization
        context = self.ln(context)
        
        # Hidden layer 1
        x = self.relu(self.fc1(context))
        x = self.dropout1(x)

        # Hidden layer 2
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output
        pred = self.output(x)
            
        return pred