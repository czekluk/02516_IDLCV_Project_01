from trainer import Trainer
from data.make_dataset import HotdogNotHotDog_DataModule
from data.custom_transforms import base_transform, random_transform
import torch
import json
import os
from models.pytorch_models import (
    BaseResNet,
    BaseInception,
    BaseDenseNet,
    BaseVGG
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
from models.pretrained_models import (
    FrozenPretrainedResNet18,
    FrozenPretrainedDenseNet121,
    FrozenPretrainedVGG11
)
from models.final_model import (
    FinalCNN,
    FinalCNN2,
    FinalCNN3
)
from visualizer import Visualizer
from main import save_results
PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # base_experiment(epochs=25)
    # dropout_experiment(epochs=25)
    # batchnorm_experiment(epochs=25)
    # weightdecay_experiments(epochs=25)
    # dataaugmentation_experiment(epochs=25)
    # deep_experiment(epochs=25)
    # pretrained_experiment(epochs=25)
    final_experiment(epochs=100)


def final_experiment(epochs=10):
    #####################################
    # 0: Final model training
    #####################################
    train_transform = random_transform(normalize=True,size=256, rotation=True, perspective=True, random_erasing=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        FinalCNN
    ]

    description = [
        "1st_FinalCNN",
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4, "weight_decay": 1e-5}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))


def base_experiment(epochs=10):
    #####################################
    # 1: Baseline experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseDenseNet,
        BaseResNet,
        BaseVGG,
        BaseCNN
    ]

    description = [
        "Baseline_DenseNet",
        "Baseline_ResNet",
    	"Baseline_VGG",
        "Baseline_BaseCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #####################################

    # train_transform = base_transform(normalize=True,size=342)
    # test_transform = base_transform(normalize=True,size=342)
    # dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    # trainloader = dm.train_dataloader()
    # testloader = dm.test_dataloader()

    # models = [
    #     BaseInception,
    # ]

    # description = [
    #     "Baseline_Inception",
    # ]

    # optimizers = [
    #     {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    # ]

    # trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    # outputs = trainer.train()
    # save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def dropout_experiment(epochs=10):
    #####################################
    # 2: Dropout experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        DropoutCNN,
        MoreDropoutCNN
    ]

    description = [
        "DropoutExp_DropoutCNN",
        "DropoutExp_MoreDropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def batchnorm_experiment(epochs=10):
    #####################################
    # 3: Batchnorm experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BatchNormCNN,
        DropoutBatchNormCNN
    ]

    description = [
        "BatchNormExp_BatchNormCNN",
        "BatchNormExp_DropoutBatchNormCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def weightdecay_experiments(epochs=10):
    #####################################
    # 4: Weight Decay experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "WeightDecayExp_BaseCNN",
        "WeightDecayExp_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4, "weight_decay": 1e-3}},
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4, "weight_decay": 1e-4}},
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4, "weight_decay": 1e-5}},
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def dataaugmentation_experiment(epochs=10):
    #####################################
    # 5: Data Augmentation experiment
    #####################################
    train_transform = random_transform(normalize=True,size=256, horizontal=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_HorizontalFlip_BaseCNN",
        "DataAugmentationExp_HorizontalFlip_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, vertical=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_VerticalFlip_BaseCNN",
        "DataAugmentationExp_VerticalFlip_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, rotation=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_Rotation_BaseCNN",
        "DataAugmentationExp_Rotation_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, perspective=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_Perspective_BaseCNN",
        "DataAugmentationExp_Perspective_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, color_jitter=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_ColorJitter_BaseCNN",
        "DataAugmentationExp_ColorJitter_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, gaussian_blur=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_GaussianBlur_BaseCNN",
        "DataAugmentationExp_GaussianBlur_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

    #################################

    train_transform = random_transform(normalize=True,size=256, random_erasing=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        BaseCNN,
        DropoutCNN,
    ]

    description = [
        "DataAugmentationExp_RandomErasing_BaseCNN",
        "DataAugmentationExp_RandomErasing_DropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}},
    ]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def deep_experiment(epochs=10):
    #####################################
    # 6: Deeper networks experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        DeepCNN,
        DeepDropoutCNN
    ]

    description = [
        "DepthExp_DeepCNN",
        "DepthExp_DeepDropoutCNN"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

def pretrained_experiment(epochs=10):
    #####################################
    # 7: Transfer Learning experiment
    #####################################
    train_transform = base_transform(normalize=True,size=256)
    test_transform = base_transform(normalize=True,size=256)
    dm = HotdogNotHotDog_DataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        FrozenPretrainedVGG11,
        FrozenPretrainedDenseNet121,
        FrozenPretrainedResNet18
    ]

    description = [
        "FrozenPretrainedVGG11",
        "FrozenPretrainedDenseNet121",
        "FrozenPretrainedResNet18"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))

if __name__ == "__main__":
    main()