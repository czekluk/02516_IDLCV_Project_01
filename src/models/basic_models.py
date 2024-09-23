import torch
import torch.nn as nn

# Model 1: Basic CNN
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Basic CNN with 2 convolutional layers and 1 fully connected layer
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(32*64*64, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 2: CNN with Dropout
class CNNWithDropout(nn.Module):
    def __init__(self):
        super(CNNWithDropout, self).__init__()
        # Added Dropout to prevent overfitting
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),  # Increased kernel size to 5
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),  # Dropout layer added here
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5)  # Dropout layer added here
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64*32*32, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 3: CNN with Batch Normalization
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super(CNNWithBatchNorm, self).__init__()
        # Added Batch Normalization to improve training stability and performance
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),  # Increased kernel size to 5
            nn.BatchNorm2d(32),  # BatchNorm layer added here
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),  # BatchNorm layer added here
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64*32*32, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 4: CNN with More Conv Layers
class CNNWithMoreConvLayers(nn.Module):
    def __init__(self):
        super(CNNWithMoreConvLayers, self).__init__()
        # Increased depth of the network to capture more complex features
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 11, padding=5),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),  # Additional convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(256*7*7, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 5: CNN with More Filters
class CNNWithMoreFilters(nn.Module):
    def __init__(self):
        super(CNNWithMoreFilters, self).__init__()
        # Increased width of the network to capture more features in each layer
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 1024, 3, padding=1),  # Increased number of filters
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(1024*32*32, 1),  # Adjusted input size
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 6: CNN with More Dense Layers
class CNNWithMoreDenseLayers(nn.Module):
    def __init__(self):
        super(CNNWithMoreDenseLayers, self).__init__()
        # Added more dense layers to improve learning capacity
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64*32*32, 128),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(128, 64),  # Additional dense layer added here
            nn.ReLU(),
            nn.Linear(64, 64),  # Additional dense layer added here
            nn.ReLU(),
            nn.Linear(64, 64),  # Additional dense layer added here
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 7: CNN with Different Activation Functions
class CNNWithDifferentActivations(nn.Module):
    def __init__(self):
        super(CNNWithDifferentActivations, self).__init__()
        # Experimented with different activation functions to improve learning
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),  # Changed activation function to LeakyReLU
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64*32*32, 128),  # Adjusted input size
            nn.LeakyReLU(0.1),  # Changed activation function to LeakyReLU
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 8: CNN with All Regularizations
class CNNWithAllRegularizations(nn.Module):
    def __init__(self):
        super(CNNWithAllRegularizations, self).__init__()
        # Combined Dropout, Batch Normalization, and Global Average Pooling
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5)
         
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64*32*32, 1),
            nn.Dropout1d(0.5),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

# Model 9: Final Model with All Improvements and average pooling
class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        # Combined all improvements: deeper, wider, dropout, batch norm, global avg pooling
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),  # Additional convolutional layer
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(256*8*8, 1024),  # Adjusted input size due to additional layer and image dimension
            nn.LeakyReLU(0.1),  # Changed activation function to LeakyReLU
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 64),  # Additional dense layer added here
            nn.ReLU(),
            nn.Linear(64, 64),  # Additional dense layer added here
            nn.LeakyReLU(0.1),  
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
