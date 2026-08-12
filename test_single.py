# test_single.py
# Simple test to verify everything works

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

print("="*60)
print("🧪 SINGLE EXPERIMENT TEST")
print("="*60)

# Test 1: Check PyTorch
print("\n✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

# Test 2: Try downloading MNIST
print("\n📥 Testing MNIST download...")
try:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Try to download MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    print("✅ MNIST downloaded successfully!")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create a simple model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model, optimizer, loss
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Train for 1 epoch
    print("\n🔄 Training for 1 epoch...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 5:  # Only use first 5 batches for quick test
            break
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"   Batch {batch_idx+1}/6, Loss: {loss.item():.4f}")
    
    print("\n✅ Test completed successfully!")
    print("The system is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Solutions:")
    print("1. Check your internet connection")
    print("2. Try running: pip install --upgrade torchvision")
    print("3. If download keeps failing, use: DATASET = 'mnist' in config.py")