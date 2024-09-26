from torchvision import models
import torch.nn as nn
import torch

class BaseResNet(nn.Module):
    # ResNet18
    # number of parameters 11.689.512
    def __init__(self, num_classes: int = 2):
        """
        Initializes the pre-trained ResNet18Model with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(BaseResNet, self).__init__()

        self.model = models.resnet18(progress=True)

        num_features = self.model.fc.in_features
        if num_classes == 2:
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        else:
            self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class BaseVGG(nn.Module):
    # VGG 11	
    # 132.863.336
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes the pretrained VGG16 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
            num_hidden_units (int): Number of hidden units in the fully-connected layer.
        """
        super(BaseVGG, self).__init__()

        self.model = models.vgg11(progress=True)

        # Numbers for linear input dimensions taken from the docs https://pytorch.org/vision/0.9/_modules/torchvision/models/vgg.html#vgg16_bn
        if num_classes == 2:
            self.model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, num_hidden_units),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, num_hidden_units),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, 1),
                nn.Sigmoid()
            )
        else:
            self.model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, num_hidden_units),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, num_hidden_units),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, num_classes),
            )

    def forward(self, x):
        return self.model(x)

class BaseDenseNet(nn.Module):
    # DenseNet 121
    # 7.978.856
    def __init__(self, num_classes: int = 2):
        """
        Initializes the pre-trained DenseNet121 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(BaseDenseNet, self).__init__()

        self.model = models.densenet121(progress=True)

        num_features = self.model.classifier.in_features
        if num_classes == 2:
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        else:
            self.model.classifier = nn.Linear(num_features, num_classes)


    def forward(self, x):
        return self.model(x)

class BaseInception(nn.Module):
    # Inception V3
    # 	27.161.264
    def __init__(self, num_classes: int = 2):
        """
        Initializes the pre-trained DenseNet121 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(BaseInception, self).__init__()

        self.model = models.inception_v3(progress=True)

        num_features = self.model.fc.in_features
        if num_classes == 2:
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        else:
            self.model.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        output = self.model(x)
        return output[0]