# model.py
# Flexible neural network with configurable layers, BatchNorm, and Dropout

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleNN(nn.Module): 
    """
    Flexible neural network architecture
    
    Parameters:
        - hidden_sizes: List of hidden layer sizes (e.g., [128, 256, 128])
        - use_batchnorm: Enable/disable Batch Normalization
        - dropout_rate: Dropout rate (0 means disabled)
        - activation: Activation function type ('relu', 'tanh', 'sigmoid')
    """
    
    def __init__(self, input_size, num_classes, hidden_sizes, 
                 use_batchnorm=False, dropout_rate=0.0, activation='relu'):
        super(FlexibleNN, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # BatchNorm (after linear, before activation)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation function
            layers.append(self.activation)
            
            # Dropout (after activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f'Activation {name} not supported')
    
    def forward(self, x):
        # Flatten input (for image data)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_config(self):
        """Return model configuration for logging"""
        return {
            'hidden_sizes': self.network[0].out_features,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate,
            'activation': str(self.activation)
        }