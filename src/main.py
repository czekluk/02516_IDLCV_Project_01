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
    FrozenPretrainedAlexNet,
    UnfrozenPretrainedAlexNet,
    FrozenPretrainedVGG,
    UnfrozenPretrainedVGG,
    FrozenPretrainedDenseNet121,
    UnfrozenPretrainedDenseNet121,
)
from models.basic_models import (
    WithoutMaxPoolCNN,
    BaseCNN,
    BatchNormCNN,
    DropoutCNN,
    MoreDropoutCNN,
    DropoutBatchNormCNN,
    DeepCNN,
    DeepDropoutCNN
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
               os.path.join(saved_models_path, 
                            f"{best_model['description']}-{best_model['timestamp'].year}-{best_model['timestamp'].month}-{best_model['timestamp'].day}_{best_model['timestamp'].hour}-{best_model['timestamp'].minute}-{best_model['timestamp'].second}-{best_model['test_acc'][-1]:.4f}-{best_model['model_name']}.pth"))
    
    # save the results
    data= []
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except:
            data = []

    # stringify the model and optimizer from the outputs as they are not json serializable
    for output in outputs:
        if "model" in output:
            output["model"] = str(output["model"])
        if "optimizer" in output["optimizer_config"]:
            output["optimizer_config"]["optimizer"] = str(output["optimizer_config"]["optimizer"])
        if "transform" in output:
            output["transform"] = str(output["transform"])
        if "timestamp" in output:
            output["timestamp"] = str(output["timestamp"])
    
    data.extend(outputs)
    # sort the entries
    data = sorted(data, key=lambda x: x['test_acc'][-1], reverse=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    train_transform = random_transform(rotation=True, rotation_degree=30,normalize=True,size=256)
    # train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()
    
    models = [
    FrozenPretrainedResNet34, #Lukas
    FrozenPretrainedVGG, #Lukas
    FrozenPretrainedDenseNet121, #Filip

    WithoutMaxPoolCNN,
    BaseCNN,
    BatchNormCNN,
    DropoutCNN,
    MoreDropoutCNN,
    DropoutBatchNormCNN,
    DeepCNN,
    DeepDropoutCNN
]
    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3, "weight_decay": 1e-4}},
        {"optimizer": torch.optim.SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
    ]
    epochs = [20]
    
    trainer = Trainer(models, optimizers, epochs, trainloader, testloader)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))
    
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments.json")
    #save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    # visualizer.plot_results(json_path=json_path)

if __name__ == "__main__":
    main()
