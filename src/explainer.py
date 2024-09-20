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
import cv2

class SaliencyExplainer:
    # https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.prepare_model()

    def prepare_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def explain(self, image: torch.Tensor):
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
        X = transform(image)

        self.model.eval()

        X.requires_grad_()

        scores = self.model(X)

        score_max_idx = scores.argmax()
        score_max = scores[0,score_max_idx]

        score_max.backward()

        saliency, _ = torch.max(X.grad.data.abs(), dim=1)

        return saliency
    
    def show_image(self, saliency: torch.Tensor, overlay: bool = False, image: torch.Tensor = torch.empty((224,224))):
        if overlay == False:
            plt.imshow(saliency[0], cmap="hot")
        else:
            saliency = saliency.numpy()
            saliency = saliency[0]*255
            saliency = saliency.astype(np.uint8)
            saliency_heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            print(saliency_heatmap.shape)
            image = image.resize((saliency.shape[0],saliency.shape[1]))
            image = np.array(image)
            result = cv2.addWeighted(saliency_heatmap,0.5,image,0.5,0)
            print(result.shape)
            plt.imshow(result)
        plt.axis('off')
        plt.show()

    def save_image(self, saliency: torch.Tensor, name: str = "sal_map_01.png"):
        # transform saliency map to greyscale image
        sal_np = saliency.numpy()
        sal_np = sal_np[0]*255
        sal_np = sal_np.astype(np.uint8)
        # apply histogram equalisation to increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        sal_np = clahe.apply(sal_np)
        # save image
        sal_img = Image.fromarray(sal_np)
        sal_img.save("results/saved_saliency_maps/" + name)

if __name__ == "__main__":
    model = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    explainer = SaliencyExplainer(model)

    img = Image.open('data/test/nothotdog/pets (632).jpg')

    result = explainer.explain(img)

    explainer.show_image(result, True, img)
    explainer.save_image(result)

    # make it independent of PIL loaded images!!!
    # -> just plain numpy image data

    


    

