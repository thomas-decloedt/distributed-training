from pathlib import Path

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.001
    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
