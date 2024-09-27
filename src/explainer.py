import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform
from torchvision.models import VGG19_Weights
from models.basic_models import BaseCNN
from models.final_model import FinalCNN3
import cv2
import os

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SaliencyExplainer:
    # https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.prepare_model()

    def prepare_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def explain(self, image: torch.Tensor, resize=False, normalize=False, size=256):
        """ Explains classification of an image using saliency maps

        Args:
            image (torch.Tensor): Image to be explained by saliency maps. Has to be of format (B,C,H,W)
            resize (bool, optional, Default: False): Enables resizing of image to input dimension of model.
            normalize (bool, optional, Default: False): Enables normalization of image.
            size (int, optional, Default: 256): SIze used for resizing.
        
        Returns:
            saliency_map (torch.Tensor): Computed saliency map
        """
        # check for correct input datatypes
        assert torch.is_tensor(image) == True
        assert len(image.shape) == 4

        # apply transform if needed
        if resize == True:
            resize_transform = T.Resize((size,size))
            image = resize_transform(image)

        if normalize == True:
            norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = norm_transform(image)

        # calculate saliency map
        self.model.eval()

        image.requires_grad_()

        scores = self.model(image)

        # get class with highest prediction
        score_max_idx = scores.argmax()
        score_max = scores[0,score_max_idx]

        # backward propagate the gradient
        score_max.backward()

        # get maximum gradient per channel
        saliency_map, _ = torch.max(image.grad.data.abs(), dim=1)

        return saliency_map
    
    def explain_smoothgrad(self, image: torch.Tensor, resize: bool = False, normalize: bool = False, size=256, n_avg=20, std=0.05):
        # check for correct input datatypes
        assert torch.is_tensor(image) == True
        assert len(image.shape) == 4

        # apply transform if needed
        if resize == True:
            resize_transform = T.Resize((size,size))
            image = resize_transform(image)

        if normalize == True:
            norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = norm_transform(image)

        # calculate saliency map
        self.model.eval()

        image_original = image
        mean = 0

        for idx in range(n_avg):
            image = image_original + torch.randn(image_original.size()) * std + mean

            image.requires_grad_()

            scores = self.model(image)

            score_max_idx = scores.argmax()
            score_max = scores[0,score_max_idx]

            score_max.backward()
            if idx == 0:
                saliency_map, _ = torch.max(image.grad.data.abs(), dim=1)
            else:
                saliency_map_temp, _ = torch.max(image.grad.data.abs(), dim=1)
                saliency_map += saliency_map_temp

        return saliency_map/n_avg
    
    def show_image(self, saliency_map: torch.Tensor, overlay: bool = False, 
                   image: torch.Tensor = torch.empty((128,128)),
                   hist_eq: bool = False, 
                   save=False, path="results/saved_saliency_maps/sal_map_01.png"):
        """ Displays saliency map.

        Args:
            saliency_map (torch.Tensor): Slaiency map to be visualized
            overlay (bool, optional, Default: False): Defines if saliency map shall be overlayed on top of image.
            normalize (torch.Tensor, optional, Default: torch.empty((128,128))): Enables normalization of image.
            save (bool, optional, Default: False): Defines if plot shall be saved.
            path (string, optional, Default: "results/saved_saliency_maps/sal_map_01.png"): Path where the figure shall be saved.
            hist_eq (bool, optional, Default: False): Defines if saliency map shall be subject to histogramm equalisation.
        """
        # convert to uint8-greyscale
        saliency_map = saliency_map.numpy()
        saliency_map = saliency_map[0]*255
        saliency_map = saliency_map.astype(np.uint8)

        # do histogram equalisation
        if hist_eq == True:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            saliency_map = clahe.apply(saliency_map)

        if overlay == False:
            # create plot
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(saliency_map, cmap="hot")
            ax1.set_title('Saliency Map')
            ax1.axis('off')
            ax2.imshow(image[0].permute(1,2,0))
            ax2.set_title('Original Image')
            ax2.axis('off')

            # save the image
            if save == True:
                plt.savefig(path, dpi=150)

            # visualize plot
            plt.show()

        else:
            # calculate heatmap
            saliency_map = 255 - saliency_map
            saliency_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
            image = image[0].permute(1,2,0).numpy()
            image = cv2.resize(image,(saliency_map.shape[0],saliency_map.shape[1]))
            image = image * 255
            image = image.astype(np.uint8)
            heatmap = cv2.addWeighted(saliency_heatmap,0.5,image,0.5,0)

            # create plot
            fig, ax = plt.subplots()
            ax.imshow(heatmap)
            ax.set_title('Saliency Heatmap')
            ax.axis('off')

            # save the image
            if save == True:
                plt.savefig(path, dpi=150)

            # display the image
            plt.show()

if __name__ == "__main__":
    model = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    # model = BaseCNN()
    # model_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models/Baseline_BaseCNN-2024-9-26_11-0-35-0.7444-BaseCNN.pth")
    # model.load_state_dict(torch.load(model_path, weights_only=True))

    model = FinalCNN3()
    model_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models/3rd_FinalCNN-2024-9-27_20-0-26-0.8303-FinalCNN3.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    test_transform = base_transform(normalize=False, size=256)
    dm = HotdogNotHotDog_DataModule(test_transform=test_transform, batch_size=10)
    testloader = dm.test_dataloader()

    img, _ = next(iter(testloader))
    img = img[4,:,:,:]
    img = torch.unsqueeze(img, 0)
    print(model(img))
    explainer = SaliencyExplainer(model)
    sal_map = explainer.explain_smoothgrad(img,normalize=True, n_avg=25, std=0.1)
    # sal_map = explainer.explain(img,normalize=True)

    # Normalize saliency map to the interval [0, 1]
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min())

    explainer.show_image(sal_map, image=img)
    explainer.show_image(sal_map, overlay=True, image=img, hist_eq=True)
    
