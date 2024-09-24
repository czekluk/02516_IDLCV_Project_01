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
    def rank_models(self, json_path=os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json"), save_path="results/ranked_models.json"):
        """Rank the models based on the final test accuracy and save the ranking to a json file
        
        Args:
            json_path (str, optional): Path to the json file containing the results. Defaults to "results/experiments_to_plot.json".
            save_path (str, optional): Path to save the ranked models json file. Defaults to "results/ranked_models.json".
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        ranked_data = sorted(data, key=lambda x: x["test_acc"][-1], reverse=True)
        
        ranked_models = [{"model_name": entry["model_name"], "final_test_acc": entry["test_acc"][-1]} for entry in ranked_data]
        
        with open(save_path, "w") as f:
            json.dump(ranked_models, f, indent=4)
        
        print(f"Ranked models saved to {save_path}")

     
if __name__ == "__main__":
       # Example usage
    
    visualizer = Visualizer()
    visualizer.rank_models( save_path=os.path.join(PROJECT_BASE_DIR, "results/ranked_models.json"))
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    visualizer.plot_results(json_path=json_path, save_path=save_path)