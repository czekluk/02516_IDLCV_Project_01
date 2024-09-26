from torchvision import models
import torch.nn as nn
import torch
from torchvision.models import ResNet18_Weights, AlexNet_Weights, VGG16_BN_Weights, DenseNet121_Weights, \
    ResNet34_Weights, VGG11_Weights

"""
This file contains different pretrained models. Each model has its base class with
a "fine_tuning" parameter, and two classes that inherit the base class and on their initialization
specify the mode (fine_tuning parameter value).
Since the Trainer initializes a model with no arguments passed, seperate classes
are defined for models with the same architecture and different modes.
"""


class PretrainedResNet18(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2):
        """
        Initializes the pre-trained ResNet18Model with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(PretrainedResNet18, self).__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

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


class FrozenPretrainedResNet18(PretrainedResNet18):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained ResNet18 model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedResNet18, self).__init__(
            fine_tuning=True,
            num_classes=num_classes
        )


class UnfrozenPretrainedResNet18(PretrainedResNet18):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained ResNet18 model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedResNet18, self).__init__(
            fine_tuning=False,
            num_classes=num_classes
        )


class PretrainedResNet34(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2):
        """
        Initializes the pre-trained ResNet34Model with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(PretrainedResNet34, self).__init__()

        self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

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


class FrozenPretrainedResNet34(PretrainedResNet34):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained ResNet34 model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedResNet34, self).__init__(
            fine_tuning=True,
            num_classes=num_classes
        )


class UnfrozenPretrainedResNet34(PretrainedResNet34):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained ResNet34 model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedResNet34, self).__init__(
            fine_tuning=False,
            num_classes=num_classes
        )


class PretrainedAlexNet(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes the pretrained AlexNet with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
            num_hidden_units (int): Number of hidden units in the fully-connected layer.
        """
        super(PretrainedAlexNet, self).__init__()

        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

        # Numbers for input linear dimension taken from the official docs:
        # https://pytorch.org/vision/0.9/_modules/torchvision/models/alexnet.html#alexnet
        if num_classes == 2:
            self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, num_hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, num_hidden_units),
                nn.ReLU(inplace=True),
                nn.Linear(num_hidden_units, 1),
                nn.Sigmoid()
            )
        else:
            self.model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, num_hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(num_hidden_units, num_hidden_units),
                nn.ReLU(inplace=True),
                nn.Linear(num_hidden_units, num_classes),
            )


    def forward(self, x):
        return self.model(x)


class FrozenPretrainedAlexNet(PretrainedAlexNet):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained AlexNet model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedAlexNet, self).__init__(
            fine_tuning=True,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


class UnfrozenPretrainedAlexNet(PretrainedAlexNet):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained AlexNet model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedAlexNet, self).__init__(
            fine_tuning=False,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


class PretrainedVGG(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes the pretrained VGG16 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
            num_hidden_units (int): Number of hidden units in the fully-connected layer.
        """
        super(PretrainedVGG, self).__init__()

        self.model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

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


class FrozenPretrainedVGG(PretrainedVGG):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained VGG16 model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedVGG, self).__init__(
            fine_tuning=True,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


class UnfrozenPretrainedVGG(PretrainedVGG):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained VGG16 model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedVGG, self).__init__(
            fine_tuning=False,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


class PretrainedDenseNet121(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2):
        """
        Initializes the pre-trained DenseNet121 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(PretrainedDenseNet121, self).__init__()

        self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

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


class FrozenPretrainedDenseNet121(PretrainedDenseNet121):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained DenseNet121 model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedDenseNet121, self).__init__(
            fine_tuning=True,
            num_classes=num_classes
        )


class UnfrozenPretrainedDenseNet121(PretrainedDenseNet121):
    def __init__(self, num_classes: int = 2):
        """
        Initializes pre-trained DenseNet121 model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedDenseNet121, self).__init__(
            fine_tuning=False,
            num_classes=num_classes
        )

class PretrainedVGG11(nn.Module):
    def __init__(self, fine_tuning: bool = False, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes the pretrained VGG16 with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
            num_hidden_units (int): Number of hidden units in the fully-connected layer.
        """
        super(PretrainedVGG11, self).__init__()

        self.model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

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


class FrozenPretrainedVGG11(PretrainedVGG11):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained VGG16 model in the fine-tuning mode, where the convolutional layers are frozen.
        """
        super(FrozenPretrainedVGG11, self).__init__(
            fine_tuning=True,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


class UnfrozenPretrainedVGG11(PretrainedVGG11):
    def __init__(self, num_classes: int = 2, num_hidden_units: int = 4096):
        """
        Initializes pre-trained VGG16 model in non fine-tuning mode, where the convolutional layers are NOT frozen and can
        be further trained.
        """
        super(UnfrozenPretrainedVGG11, self).__init__(
            fine_tuning=False,
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )