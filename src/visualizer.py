import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import os

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
        
if __name__ == "__main__":
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    visualizer.plot_results(json_path=json_path, save_path=save_path)