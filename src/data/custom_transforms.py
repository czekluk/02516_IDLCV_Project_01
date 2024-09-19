import torchvision.transforms as transforms


def base_transform(size: int = 128, normalize: bool = False):
    """Base transformation for the image - resize and convert to tensor.
    If normalize is True, normalize the image with the ImageNet mean and standard deviation
    (https://pytorch.org/vision/0.9/transforms.html)
    """
    transform_list = [transforms.Resize((size, size)), transforms.ToTensor()]
    if normalize:
        transform_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
    transform = transforms.Compose(transform_list)
    return transform


def random_transform(
    size: int = 128,
    normalize: bool = False,
    horizontal: bool = False,
    horizontal_p: float = 0.5,
    vertical: bool = False,
    vertical_p: float = 0.5,
    rotation: bool = False,
    rotation_degree: int = 30,
    perspective: bool = False,
    perspective_p: float = 0.5,
):
    """Random transform chain for the image - resize, random horizontal flip, random vertical flip, random rotation, etc."""
    transform_list = [transforms.Resize((size, size))]

    if horizontal:
        transform_list.append(transforms.RandomHorizontalFlip(horizontal_p))
    if vertical:
        transform_list.append(transforms.RandomVerticalFlip(vertical_p))
    if perspective:
        transform_list.append(transforms.RandomPerspective(p=perspective_p))
    if rotation:
        transform_list.append(transforms.RandomRotation(rotation_degree))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    transform = transforms.Compose(transform_list)
    return transform
