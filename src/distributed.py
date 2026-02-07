import os

import torch
import torch.distributed as dist

from src.config import DistributedConfig


def get_distributed_config() -> DistributedConfig:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedConfig(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
    )


def setup_distributed(config: DistributedConfig) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(
        backend=config.backend,
        rank=config.rank,
        world_size=config.world_size,
    )


def cleanup_distributed() -> None:
    dist.destroy_process_group()
