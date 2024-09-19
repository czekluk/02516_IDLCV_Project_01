from trainer import Trainer, DummyNet
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform
import torch
import torch.nn as nn


def main():
    train_transform = base_transform()
    test_transform = base_transform()
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()
    
    models = [DummyNet, DummyNet]
    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}},
        {"optimizer": torch.optim.SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
    ]
    epochs = [2]
    
    trainer = Trainer(models, optimizers, epochs, trainloader, testloader)
    outputs = trainer.train()
    print(outputs)

if __name__ == "__main__":
    main()
