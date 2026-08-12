# trainer_fixed.py
# Simplified version - uses only MNIST (most stable)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
import time

from model import FlexibleNN

class Trainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, optimizer, criterion, epochs):
        print(f"Training on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')
        
        return self.history
    
    def evaluate(self, test_loader):
        return self.validate(test_loader, nn.CrossEntropyLoss())

def run_experiment(config, model_config):
    """Run a complete experiment"""
    
    # Set seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print(f"\n📥 Loading MNIST dataset...")
    
    try:
        # Only use MNIST for stability
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        input_size = 28 * 28
        
    except Exception as e:
        print(f"❌ Error loading MNIST: {e}")
        print("💡 Make sure you have internet connection for first-time download")
        raise
    
    # Split train/validation
    train_size = int((1 - config.VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = FlexibleNN(
        input_size=input_size,
        num_classes=10,
        hidden_sizes=model_config['hidden_sizes'],
        use_batchnorm=model_config['use_batchnorm'],
        dropout_rate=model_config['dropout_rate'],
        activation='relu'
    ).to(device)
    
    # Optimizer
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train
    trainer = Trainer(model, device, config)
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, optimizer, criterion, config.EPOCHS)
    train_time = time.time() - start_time
    
    # Evaluate
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")
    
    return {
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_time': train_time,
        'model_config': model_config
    }  