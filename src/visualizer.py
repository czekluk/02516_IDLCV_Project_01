import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from datetime import datetime
import os
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Visualizer():
    def __init__(self):
        pass
    
    def plot_results(self, json_path="results/experiments_to_plot.json", cmap='Spectral', save_path=None, figsize=(8, 5)):
        """Plot the results of the experiments
        
        Args:
            json_path (str, optional): Path to the json file containing the results. Defaults to "results/experiments_to_plot.json".
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        colors = cm.get_cmap(cmap, len(data))
        
        plt.figure(figsize=figsize)
        for i, entry in enumerate(data):
            plt.plot(entry["train_acc"], label=f"{entry['model_name']} train acc", linestyle="--", color=colors(i))
            plt.plot(entry["test_acc"], label=f"{entry['model_name']} test acc", color=colors(i))
        plt.xlabel("Epoch")
        plt.xticks(range(max([len(entry["train_acc"]) for entry in data])))
        plt.ylabel("Accuracy")
        plt.legend()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"plot_{time}.png")
            
        plt.show()

    def plot_images(self, data: torch.Tensor, grid_height: int = 4, grid_width: int = 4):
        max_img = grid_height * grid_width
        data = data[0:max_img]
        if grid_height == grid_width/2:
            fig, axs = plt.subplots(grid_height, grid_width, figsize=(15,7.5))
        else:
            fig, axs = plt.subplots(grid_height, grid_width, figsize=(15,15))
        counter = 0
        for idx in range(grid_height):
            for idy in range(grid_width):
                img = data[counter]
                img = img.permute((1,2,0)).numpy()
                axs[idx, idy].imshow(img)
                axs[idx, idy].axis('off')
                counter += 1

        plt.tight_layout()
        plt.show()

        
if __name__ == "__main__":
    visualizer = Visualizer()
    # json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    # save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    # visualizer.plot_results(json_path=json_path, save_path=save_path)

    train_transform = base_transform()
    test_transform = base_transform()
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.train_dataloader()

    img, _ = next(iter(trainloader))
    # img, _ = next(iter(testloader))

    visualizer.plot_images(img, grid_height=2)

    dm.plot_examples()