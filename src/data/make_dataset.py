import os
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import data.custom_transforms

from torch.utils.data import DataLoader, Dataset

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_BASE_DIR, "data")


class HotdogNotHotdog_Dataset(Dataset):
    def __init__(
        self, train: bool, transform=transforms.ToTensor(), data_path=DATA_DIR
    ):
        "Initialization"
        self.transform = transform
        data_path = os.path.join(data_path, "train" if train else "test")
        image_classes = [
            os.path.split(d)[1] for d in glob.glob(data_path + "/*") if os.path.isdir(d)
        ]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + "/*/*.jpg")

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


class HotdogNotHotDog_DataModule:
    def __init__(
        self,
        data_path=DATA_DIR,
        batch_size: int = 64,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor(),
    ):
        """Custom data module class for the HotdogNotHotdog dataset. Used for
        loading of data, train/test splitting and constructing dataloaders.

        Args:
            data_path: Path to the data directory. Default: (Project_Base_Dir)/data
            batch_size (int, optional): Batch size for the dataloaders. Defaults to 64.
            train_transform (_type_, optional): Transform to apply to train data. Defaults to transforms.ToTensor().
            test_transform (_type_, optional): Transform to apply to test data. Defaults to transforms.ToTensor().
        """
        assert type(data_path)==str, "data_path needs to be a string"
        self.batch_size = batch_size
        self.train_dataset = HotdogNotHotdog_Dataset(
            train=True, transform=train_transform, data_path=data_path
        )
        self.test_dataset = HotdogNotHotdog_Dataset(
            train=False, transform=test_transform, data_path=data_path
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
        """Return the training dataloader

        Args:
            shuffle (bool, optional, Default: True): Whether to shuffle the dataset. Defaults to True.

        Returns:
            DataLoader: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,
        )

    def test_dataloader(self, shuffle=False):
        """Return the test dataloader

        Args:
            shuffle (bool, optional, Default: False): Whether to shuffle the dataset. Defaults to True.

        Returns:
            DataLoader: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,
        )

    def get_training_examples(self):
        """Return the first batch of training examples

        Returns:
            images: torch.Tensor
            labels: torch.Tensor
        """
        images, labels = next(iter(trainloader))
        return images, labels

    def get_test_examples(self):
        """Return the first batch of test examples

        Returns:
            images: torch.Tensor
            labels: torch.Tensor
        """
        images, labels = next(iter(testloader))
        return images, labels

    def plot_examples(self):
        """Plot the first batch of training examples"""
        images, labels = next(iter(trainloader))

        plt.figure(figsize=(20, 10))

        for i in range(self.batch_size // 2):
            plt.subplot(self.batch_size // 3 + 1, self.batch_size // 3, i + 1)
            plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
            plt.title(["hotdog", "not hotdog"][labels[i].item()])
            plt.axis("off")

    def __repr__(self):
        return (
            f"HotdogNotHotDog DataModule with batch size {self.batch_size}\n"
            + f" Training dataset: {len(self.train_dataset)} samples\n"
            + f" Test dataset: {len(self.test_dataset)} samples"
        )
    
    def get_trainset_size(self):
        return len(self.train_dataset)

    def get_testset_size(self):
        return len(self.test_dataset)


if __name__ == "__main__":
    print("Data directory: ", DATA_DIR)

    img_size = 128
    train_transform = custom_transforms.random_transform(
        size=img_size, horizontal=True, vertical=True, rotation=True, normalize=False
    )
    test_transform = custom_transforms.base_transform(size=img_size, normalize=False)

    dm = HotdogNotHotDog_DataModule(
        train_transform=train_transform, test_transform=test_transform
    )
    print(dm)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    images, labels = next(iter(trainloader))
    plt.figure(figsize=(20, 10))

    for i in range(21):
        plt.subplot(5, 7, i + 1)
        plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
        plt.title(["hotdog", "not hotdog"][labels[i].item()])
        plt.axis("off")
    
    plt.show()
