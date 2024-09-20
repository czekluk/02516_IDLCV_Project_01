import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform
from typing import List
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights
from trainer import DummyNet, Trainer
from PIL import Image

class SaliencyExplainer:
    # https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.prepare_model()

    def prepare_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def explain(self, image):
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
        X = transform(image)

        model.eval()

        X.requires_grad_()

        scores = model(X)

        score_max_idx = scores.argmax()
        score_max = scores[0,score_max_idx]

        score_max.backward()

        saliency, _ = torch.max(X.grad.data.abs(), dim=1)

        return saliency


if __name__ == "__main__":
    model = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    explainer = SaliencyExplainer(model)

    img = Image.open('data/test/nothotdog/pets (632).jpg')

    result = explainer.explain(img)

    plt.imshow(result[0], cmap="hot")
    plt.axis('off')
    plt.show()


    

