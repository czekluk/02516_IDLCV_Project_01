import torch
import torch.nn as nn



class WithoutMaxPoolCNN(nn.Module):
    def __init__(self):
        super(WithoutMaxPoolCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.ReLU(), 
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(256*256*256, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
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



class BatchNormCNN(nn.Module):
    def __init__(self):
        super(BatchNormCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.BatchNorm2d(256),
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



# Best model so far- 78.73
class DropoutCNN(nn.Module):
    def __init__(self):
        super(DropoutCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.05),
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
    
# Best model so far- 78.73
class MoreDropoutCNN(nn.Module):
    def __init__(self):
        super(MoreDropoutCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.2),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.2),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.2),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.2),
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
    

class DropoutBatchNormCNN(nn.Module):
    def __init__(self):
        super(DropoutBatchNormCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(256),
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
    


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
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
    



class DeepDropoutCNN(nn.Module):
    def __init__(self):
        super(DeepDropoutCNN, self).__init__()
        self.convolutional = nn.Sequential(
            # Input: 3x256x256
            # Layer 1
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 32x128x128
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            
            # Pooling layer: 64x64x64
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            
            # Pooling layer: 128x32x32
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.Dropout2d(0.05),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
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