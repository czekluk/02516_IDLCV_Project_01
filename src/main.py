from trainer import Trainer
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform, random_transform
import torch
import json
import os
from models.pretrained_models import (
    FrozenPretrainedResNet18,
    UnfrozenPretrainedResNet18,
    FrozenPretrainedResNet34,
    UnfrozenPretrainedResNet34,
    FrozenPretrainedResNet101,
    UnfrozenPretrainedResNet101,
    FrozenPretrainedAlexNet,
    UnfrozenPretrainedAlexNet,
    FrozenPretrainedVGG,
    UnfrozenPretrainedVGG,
    FrozenPretrainedDenseNet121,
    UnfrozenPretrainedDenseNet121,
)
from models.basic_models import (
    BasicCNN,
    CNNWithDropout,
    CNNWithBatchNorm,
    CNNWithMoreConvLayers,
    CNNWithMoreFilters,
    CNNWithMoreDenseLayers,
    CNNWithDifferentActivations,
    CNNWithAllRegularizations,
    FinalModel
)
from visualizer import Visualizer
PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_results(outputs, path=os.path.join(PROJECT_BASE_DIR, "results/experiments.json")):
    """
    Saves the best model from outputs (the parameters).
    Saves all of the results into a results/experiments.json
    """
    saved_models_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models")
    # save the best model
    best_model = outputs[0]
    torch.save(best_model["model"].state_dict(), 
               os.path.join(saved_models_path, f"{best_model['test_acc'][-1]:.4f}-{best_model['model_name']}.pth"))
    
    # save the results
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    # stringify the model and optimizer from the outputs as they are not json serializable
    for output in outputs:
        if "model" in output:
            output["model"] = str(output["model"])
        if "optimizer" in output["optimizer_config"]:
            output["optimizer_config"]["optimizer"] = str(output["optimizer_config"]["optimizer"])
    
    data.extend(outputs)
    # sort the entries
    data = sorted(data, key=lambda x: x['test_acc'][-1], reverse=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    train_transform = random_transform(horizontal=True, rotation=True, rotation_degree=30, normalize=True, size=256)
    test_transform = base_transform(size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()
    
    models = [
        FrozenPretrainedResNet101, #Lukas
        FrozenPretrainedVGG, #Lukas
        FrozenPretrainedDenseNet121, #Filip

        # BasicCNN, #Filip
        # CNNWithDropout, #Filip
        # CNNWithBatchNorm, #Alex
        # CNNWithMoreConvLayers, #Alex
        # CNNWithMoreFilters, #Zeljko
        # CNNWithMoreDenseLayers, #Zeljko
        # CNNWithDifferentActivations, #Nandor
        # CNNWithAllRegularizations, #Nandor
        # FinalModel #Nandor
    ]
    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}},
        # {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-2}},
        #{"optimizer": torch.optim.SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
    ]
    epochs = [10]
    
    trainer = Trainer(models, optimizers, epochs, trainloader, testloader)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))
    
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments.json")
    #save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    visualizer.plot_results(json_path=json_path)

if __name__ == "__main__":
    main()
