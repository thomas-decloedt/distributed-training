"""Shared DataLoader creation with DistributedSampler — used by mnist and lora."""

from collections.abc import Callable
from typing import Any

from torch.utils.data import DataLoader, Dataset


def create_distributed_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    rank: int,
    world_size: int,
    *,
    collate_fn: Callable[..., Any] | None = None,
    num_workers: int = 1,
) -> DataLoader[Any]:
    """DistributedSampler + DataLoader creation — shared by mnist and lora."""
    from torch.utils.data import DistributedSampler

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
