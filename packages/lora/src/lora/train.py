import os
import subprocess
import sys
import uuid

import torch
from common.config import VertexConfig
from common.distributed import (
    DistributedConfig,
    cleanup_distributed,
    get_distributed_config,
    setup_distributed,
)
from common.tracking import init_experiment, log_metrics, log_params, start_run
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora.config import LoRAConfig, TrainingConfig
from lora.data import get_dataloader


def run_training(
    training_config: TrainingConfig,
    lora_config: LoRAConfig | None = None,
    distributed_config: DistributedConfig | None = None,
    vertex_config: VertexConfig | None = None,
) -> None:
    cfg = distributed_config or get_distributed_config()
    setup_distributed(cfg)

    if not torch.cuda.is_available():
        threads = min(2, os.cpu_count() or 2)
        torch.set_num_threads(threads)

    device = (
        torch.device(f"cuda:{cfg.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        training_config.model_name,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
    )
    lora_cfg = lora_config or LoRAConfig(model_name=training_config.model_name)
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)
    model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    dataloader = get_dataloader(
        training_config,
        tokenizer,
        cfg.rank,
        cfg.world_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)

    training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id = ""
    checkpoint_dir = training_config.checkpoint_dir / "lora"

    if cfg.rank == 0 and vertex_config:
        init_experiment(
            VertexConfig(**{**vertex_config.model_dump(), "experiment_name": "distributed-lora"})
        )
        run_id = str(uuid.uuid4())[:8]
        start_run(f"run-{training_config.epochs}e-{training_config.batch_size}bs-{run_id}")
        log_params(
            {
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "lr": training_config.lr,
                "model": lora_cfg.model_name,
            }
        )

    for epoch in range(training_config.epochs):
        sampler = dataloader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if cfg.rank == 0:
            print(f"Epoch {epoch + 1}/{training_config.epochs} loss={avg_loss:.4f}")
            if vertex_config:
                log_metrics({"loss": avg_loss})

    if cfg.rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        peft_model = model.module
        peft_model.save_pretrained(checkpoint_dir)

    is_master = cfg.rank == 0
    cleanup_distributed()

    if is_master and vertex_config:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "common.upload_standalone",
                str(checkpoint_dir.absolute()),
                run_id,
                "lora",
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
