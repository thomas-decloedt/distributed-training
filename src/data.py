import os

from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from src.config import TrainingConfig


def get_dataloader(
    config: TrainingConfig,
    rank: int,
    world_size: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # Each rank runs in its own container/VM with no shared filesystem, so each must download.
    dataset = datasets.MNIST(
        str(config.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    num_workers = 1 if (os.cpu_count() or 1) >= 2 else 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def get_test_dataloader(config: TrainingConfig) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        str(config.data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
