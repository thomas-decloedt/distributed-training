import os

from common.dataloader import create_distributed_dataloader
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist.config import TrainingConfig


def get_dataset(config: TrainingConfig, train: bool = True) -> datasets.MNIST:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return datasets.MNIST(
        str(config.data_dir),
        train=train,
        download=True,
        transform=transform,
    )


def get_dataloader(
    config: TrainingConfig,
    rank: int,
    world_size: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = get_dataset(config, train=True)
    num_workers = 1 if (os.cpu_count() or 1) >= 2 else 0
    return create_distributed_dataloader(
        dataset,
        batch_size=config.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
    )


def get_test_dataloader(config: TrainingConfig) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = get_dataset(config, train=False)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
