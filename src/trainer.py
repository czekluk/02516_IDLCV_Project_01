import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform
from typing import List
from torch.utils.data import DataLoader


class DummyNet(nn.Module):
    """
    Placeholder network used for testing. 
    TODO delete after implementation of other models.
    """
    def __init__(self):
        super(DummyNet, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(16*64*64, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 2)
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


class Trainer:
    
    def __init__(self, models: List[nn.Module], optimizer_functions: List[dict], 
                 epochs: int, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """
        Class for training different models with different optimizers and different numbers of epochs.
        
        Args:   models              -   list of models. The models are not instances but classes. example: [AlexNet, ResNet]
                optimizer_funcitons -   list of dictionaries specifying different optimizers.
                                        example: optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}]
                epochs              -   list of different epochs to train. example: [10, 15]
                train_loader        -   torch.utils.data.DataLoader
                test_loader         -   torch.utils.data.DataLoader
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.models = models
        self.optimizer_functions = optimizer_functions
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    
    def train(self) -> List[dict]:
        """
        Train the different models, with different optimizers, with different number of epochs.
        
        Returns:    List of dictionaries representing different experiments.
                    The list is sorted in descending order based on the achieved accuracy
                    after the final epoch.
        """
        outputs = []
        for network in self.models:
            for optimizer_config in self.optimizer_functions:
                for epoch_no in self.epochs:
                    print("#########################################################")
                    print(f"Training model: {network.__name__}")
                    print(f"Optimizer: {optimizer_config['optimizer'].__name__}")
                    print(f"Training for {epoch_no} epochs")
                    model = network()
                    out_dict = self._train_single_configuration(model, optimizer_config, epoch_no)
                    outputs.append(out_dict)
        outputs_sorted = sorted(outputs, key=lambda x: x['test_acc'][-1], reverse=True)
        return outputs_sorted
    
    
    def _train_single_configuration(self, model: nn.Module, optimizer_config: dict, num_epochs: int) -> dict:
        model.to(self.device)
        optimizer = optimizer_config["optimizer"](model.parameters(), **optimizer_config["params"])
        
        out_dict = {
            'model_name':       model.__class__.__name__,
            'model':            model,
            'train_acc':        [],
            'test_acc':         [],
            'train_loss':       [],
            'test_loss':        [],
            'epochs':           num_epochs,
            'optimizer_config': optimizer_config 
            }
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            train_correct = 0
            train_loss = []
            
            for minibatch_no, (data, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                predicted = output.argmax(1)
                train_correct += (target==predicted).sum().cpu().item()
            
            test_loss = []
            test_correct = 0
            model.eval()
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                test_loss.append(self.criterion(output, target).cpu().item())
                predicted = output.argmax(1)
                test_correct += (target==predicted).sum().cpu().item()
            out_dict['train_acc'].append(train_correct/len(self.train_loader.dataset))
            out_dict['test_acc'].append(test_correct/len(self.test_loader.dataset))
            out_dict['train_loss'].append(np.mean(train_loss))
            out_dict['test_loss'].append(np.mean(test_loss))
            print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
            
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        return out_dict
            

if __name__ == "__main__":
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
    