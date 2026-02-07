import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from src.config import DistributedConfig, TrainingConfig, VertexConfig
from src.data import get_dataloader, get_test_dataloader
from src.distributed import cleanup_distributed, get_distributed_config, setup_distributed
from src.model import MnistCNN
from src.tracking import init_experiment, log_metrics, log_params, start_run


def train_one_epoch(
    model: DDP,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
) -> float:
    model.train()
    total_loss = 0.0
    for _batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(
    model: DDP,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    test_loss = total_loss / len(dataloader)
    test_accuracy = correct / total
    return test_loss, test_accuracy


def run_training(
    training_config: TrainingConfig,
    distributed_config: DistributedConfig | None = None,
    vertex_config: VertexConfig | None = None,
) -> None:
    if distributed_config is None:
        distributed_config = get_distributed_config()
    setup_distributed(distributed_config)

    if not torch.cuda.is_available():
        threads = min(2, os.cpu_count() or 2)
        torch.set_num_threads(threads)

    device = (
        torch.device(f"cuda:{distributed_config.local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = MnistCNN().to(device)
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    dataloader = get_dataloader(
        training_config,
        distributed_config.rank,
        distributed_config.world_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
    loss_fn = nn.CrossEntropyLoss()

    training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id = ""
    checkpoint_path: Path = training_config.checkpoint_dir / "model.pt"

    if distributed_config.rank == 0 and vertex_config:
        init_experiment(vertex_config)
        run_id = str(uuid.uuid4())[:8]
        start_run(f"run-{training_config.epochs}e-{training_config.batch_size}bs-{run_id}")
        log_params(
            {
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "lr": training_config.lr,
            }
        )

    for epoch in range(training_config.epochs):
        sampler = dataloader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, loss_fn, device, distributed_config.rank
        )
        if distributed_config.rank == 0:
            print(f"Epoch {epoch + 1}/{training_config.epochs} loss={avg_loss:.4f}")
            if vertex_config:
                log_metrics({"loss": avg_loss})

    if distributed_config.rank == 0:
        checkpoint_path = training_config.checkpoint_dir / "model.pt"
        torch.save(cast(MnistCNN, model.module).state_dict(), checkpoint_path)
        test_dataloader = get_test_dataloader(training_config)
        test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
        print(f"Test loss={test_loss:.4f} accuracy={test_accuracy:.4f}")
        if vertex_config:
            log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})

    is_master = distributed_config.rank == 0
    cleanup_distributed()

    if is_master and vertex_config:
        # Run upload in a detached subprocess so the main process exits quickly
        # and the torchrun exit barrier does not time out.
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.upload_standalone",
                str(checkpoint_path.absolute()),
                run_id,
            ],
            start_new_session=True,
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )


def main() -> None:
    training_config = TrainingConfig()
    vertex_config = VertexConfig.from_env()
    run_training(training_config, vertex_config=vertex_config)


if __name__ == "__main__":
    main()
