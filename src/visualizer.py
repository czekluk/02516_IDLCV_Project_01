import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns
import numpy as np
import torch
from sklearn import metrics
from data.make_dataset import HotdogNotHotDog_DataModule, base_transform
from models.basic_models import BaseCNN
from models.final_model import FinalCNN3
from tqdm import tqdm

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

def compute_confusion_matrix(target, pred, normalize=None):
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return metrics.confusion_matrix(target, pred, normalize=normalize)

def normalize(matrix, axis):
    axis = {'true': 1, 'pred': 0}[axis]
    return matrix / matrix.sum(axis=axis, keepdims=True)

def inverse_normalize(image):
    image = np.transpose(image, (1,2,0))
    return (((image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255).astype('uint8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Visualizer():
    def __init__(self):
        self.confusion_matrix = None
        self.model = None
        self.datamodule = None
        self.losses = None
        self.images = None
        self.targets = None
        self.predictions = None
    
    def plot_training_json(self, json_path, cmap='Spectral', save_path=None, figsize=(8, 5)):
        """Plot the results of the experiments
        
        Args:
            json_path (str, optional): Path to the json file containing the results. Defaults to "results/experiments_to_plot.json".
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        colors = plt.get_cmap(cmap, len(data))
        
        plt.figure(figsize=figsize)
        for i, entry in enumerate(data):
            plt.plot(entry["train_acc"], label=f"{entry['model_name']} train acc", linestyle="--", color=colors(i), linewidth=3)
            plt.plot(entry["test_acc"], label=f"{entry['model_name']} test acc", color=colors(i), linewidth=3)
        plt.xlabel("Epoch")
        max_epoch = max([len(entry["train_acc"]) for entry in data])
        plt.xticks(range(0, max_epoch, 5))
        plt.ylabel("Accuracy")
        plt.legend()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"training_{time}.png")
        plt.show()

    def plot_confusion_matrix(self, normalize = False, save_path=None):
        if self.confusion_matrix is None:
            raise Exception("Error: Model has not been evaluated yet. Please evaluate the model first using the Visualizer.evaluate_model() method.")
        data = normalize(self.confusion_matrix, axis='true') if normalize else self.confusion_matrix
        x_labels = ["Hotdog", "No Hotdog"]
        y_labels = x_labels
        plt.figure(figsize=(3, 3))        
        sns.heatmap(
            ax=plt.gca(),
            data=data,
            annot=True,
            linewidths=0.5,
            cmap="Reds",
            cbar=False,
            fmt='g',
            xticklabels=x_labels,
            yticklabels=y_labels,
            annot_kws={"size": 18}  # Increase font size
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"confmatrix_{time}.png")
        plt.show()
        
    def plot_top_k_images_with_highest_loss(self, k=3, save_path=None):
        if self.losses is None or self.images is None:
            raise Exception("Error: Model has not been evaluated yet. Please evaluate the model first using the Visualizer.evaluate_model() method.")
        top_k_indices = np.argsort(self.losses)[-k:]
        top_k_images = [self.images[i] for i in top_k_indices]
        top_k_losses = [self.losses[i] for i in top_k_indices]
        top_k_targets = [int(self.targets[i]) for i in top_k_indices]
        top_k_predictions = [int(self.predictions[i]) for i in top_k_indices]
        
        f, ax = plt.subplots(1, k, figsize=(3*k, 3))
        f.suptitle(f"Top {k} misclassified images with highest loss")
        for i, image in enumerate(top_k_images):
            ax[i].imshow(inverse_normalize(image))
            ax[i].set_title(f"Loss: {top_k_losses[i]:.2f}")
            labels = ["No Hotdog", "Hotdog"]
            ax[i].text(0.5, -0.1, f"Tar: {labels[top_k_predictions[i]]}, Pred:{labels[top_k_targets[i]]}", 
                       fontsize=8, ha='center', transform=ax[i].transAxes)
            ax[i].axis('off')
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"top_k_{i}_{time}.png")
            
        plt.show()
        
    def evaluate_model(self, model, datamodule: HotdogNotHotDog_DataModule):
        # Init parameters
        self.model = model
        self.datamodule = datamodule
        print("Evaluating model...")
        self.model.eval()
        confusion_matrix = np.zeros((2, 2))
        test_loader = datamodule.test_dataloader()
        criterion = torch.nn.BCELoss(reduction='none')
        losses = []
        images = []
        targets = []
        predictions = []
        
        # Evaluate model
        with torch.no_grad():
            for minibatch_no, (inputs, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, target.float())
                losses.extend(loss.cpu().numpy())
                images.extend(inputs.cpu().numpy())
                targets.extend(target.cpu().numpy())
                prediction = (outputs > 0.5).int()
                predictions.extend(prediction.cpu().numpy())
                confusion_matrix += compute_confusion_matrix(target, prediction)
        print("Model successfully evaluated. Confusion matrix and top k images with highest loss are available.")
        self.confusion_matrix = confusion_matrix
        self.losses = losses
        self.images = images
        self.targets = targets
        self.predictions = predictions 

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
    # Init class
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")

    # visualizer.plot_training_json(json_path=json_path, save_path=save_path, cmap='rainbow')

    # Load model
    model = FinalCNN3()
    model_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models/3rd_FinalCNN-2024-9-27_20-0-26-0.8303-FinalCNN3.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    
    # Load datamodule
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(test_transform=test_transform)
    
    # Evaluate model using class
    visualizer.evaluate_model(model=model, datamodule=dm)
    
    # Plot all results
    #visualizer.plot_training_json(json_path=json_path, save_path=save_path)
    visualizer.plot_confusion_matrix(normalize=False, save_path=save_path)
    visualizer.plot_top_k_images_with_highest_loss(k=3, save_path=save_path)

    # # Plot array of training images
    # train_transform = base_transform(normalize=True,size=256)
    # test_transform = base_transform(normalize=True,size=256)
    # dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    # trainloader = dm.train_dataloader()
    # testloader = dm.test_dataloader()

    # img, _ = next(iter(trainloader))
    # visualizer.plot_images(img, grid_height=2, grid_width=4)