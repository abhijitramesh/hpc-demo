from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from .logger_utils import get_logger
logger = get_logger()

def get_fashion_mnist_dataset(*, train=True):
    """Get Fashion-MNIST dataset.

    Args:
        train (bool, optional): If True, return training dataset. Otherwise, return test dataset. Defaults to True.

    Returns:
        torch.utils.data.Dataset: Fashion-MNIST dataset.
    """
    logger.debug(f"Featching Fashion-MNIST dataset with train as {train}...")
    return datasets.FashionMNIST(
        root="data", train=train, download=True, transform=ToTensor()
    )

def get_dataloader(*, dataset, batch_size):
    """Get dataloader for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset for which dataloader is to be created.
        batch_size (int): Batch size for dataloader.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the given dataset.
    """
    logger.debug(f"Creating dataloader for dataset with length {len(dataset)} and batch size {batch_size}...")
    return DataLoader(dataset, batch_size=batch_size)
