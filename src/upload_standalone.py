"""Upload/register model in a detached process. Avoids blocking the exit barrier after training."""

import sys
from pathlib import Path

from src.config import VertexConfig
from src.registry import register_model, upload_model
from src.tracking import end_run


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m src.upload_standalone <checkpoint_path> <run_id>")
    checkpoint_path = Path(sys.argv[1])
    run_id = sys.argv[2]
    vertex_config = VertexConfig.from_env()
    if not vertex_config:
        raise SystemExit("GOOGLE_CLOUD_PROJECT and VERTEX_STAGING_BUCKET required")
    gcs_uri = f"gs://{vertex_config.staging_bucket}/models/{run_id}/model.pt"
    upload_model(checkpoint_path, gcs_uri)
    upload_model(checkpoint_path, f"gs://{vertex_config.staging_bucket}/models/latest/model.pt")
    register_model(gcs_uri, f"mnist-{run_id}", vertex_config)
    end_run()


if __name__ == "__main__":
    main()
