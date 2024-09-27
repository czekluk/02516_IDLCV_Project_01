import torch
import torch.nn as nn

class FinalCNN(nn.Module):
    def __init__(self):
        super(FinalCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 5

            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 256x16x16
            nn.MaxPool2d(2),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(256*16*16, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
class FinalCNN2(nn.Module):
    def __init__(self):
        super(FinalCNN2, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(128, 128, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 5
            nn.Conv2d(128, 256, 3, padding=1),  
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 6

            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 7
            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 256x16x16
            nn.MaxPool2d(2),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(256*16*16, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
class FinalCNN3(nn.Module):
    def __init__(self):
        super(FinalCNN3, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 4
            nn.Conv2d(128, 128, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 5
            nn.Conv2d(128, 256, 3, padding=1),  
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 6
            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),
            
            # Pooling layer: 256x16x16
            nn.MaxPool2d(2),

            # Layer 7
            nn.Conv2d(256, 512, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Layer 8
            nn.Conv2d(512, 512, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.1),
            nn.ReLU(),

            # Pooling layer: 512x8x8
            nn.MaxPool2d(2),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(512*8*8, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x