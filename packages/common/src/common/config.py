import os
from pathlib import Path

from pydantic import BaseModel


class VertexConfig(BaseModel):
    project_id: str
    location: str = "us-central1"
    staging_bucket: str
    experiment_name: str = "distributed-mnist"

    @classmethod
    def from_env(cls) -> "VertexConfig | None":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        bucket = os.environ.get("VERTEX_STAGING_BUCKET")
        if project and bucket:
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            return cls(project_id=project, staging_bucket=bucket, location=location)
        return None


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.001
    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")


class DistributedConfig(BaseModel):
    rank: int
    world_size: int
    local_rank: int = 0
    backend: str = "gloo"
    master_addr: str = "localhost"
    master_port: int = 29500
