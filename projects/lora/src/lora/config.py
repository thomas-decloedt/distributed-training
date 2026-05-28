from pathlib import Path

from pydantic import BaseModel


class LoRAConfig(BaseModel):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = ["q_proj", "v_proj"]
    lora_dropout: float = 0.05


class TrainingConfig(BaseModel):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    epochs: int = 3
    batch_size: int = 4
    lr: float = 2e-5
    max_length: int = 512
    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
