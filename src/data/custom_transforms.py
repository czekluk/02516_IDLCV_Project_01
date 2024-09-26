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
    color_jitter: bool = False,
    brightness: float = 0.5,
    contrast: float = 0.5,
    saturation: float = 0.5,
    hue: float = 0.5,
    gaussian_blur: bool = False,
    kernel_size: int = 9,
    random_erasing: bool = False,
    random_erasing_p: float = 0.5,
):
    """Random transform chain for the image - resize, random horizontal flip, random vertical flip, random rotation, etc."""
    transform_list = [transforms.Resize((size, size))]

    transform_list.append(transforms.ToTensor())

    if color_jitter:
        transform_list.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

    if normalize:
        transform_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    if horizontal:
        transform_list.append(transforms.RandomHorizontalFlip(horizontal_p))
    if vertical:
        transform_list.append(transforms.RandomVerticalFlip(vertical_p))
    if perspective:
        transform_list.append(transforms.RandomPerspective(p=perspective_p))
    if rotation:
        transform_list.append(transforms.RandomRotation(rotation_degree))
    if gaussian_blur:
        transform_list.append(transforms.GaussianBlur(kernel_size, sigma=(0.1,2.0)))
    if random_erasing:
        transform_list.append(transforms.RandomErasing(random_erasing_p))

    transform = transforms.Compose(transform_list)
    return transform
