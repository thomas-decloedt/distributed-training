"""Upload/register model in a detached process. Avoids blocking the exit barrier after training."""

import sys
from pathlib import Path

from common.config import VertexConfig
from common.registry import register_model, upload_directory, upload_model
from common.tracking import end_run


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m common.upload_standalone <checkpoint_path> <run_id> [case=mnist]"
        )
    checkpoint_path = Path(sys.argv[1])
    run_id = sys.argv[2]
    case = sys.argv[3] if len(sys.argv) > 3 else "mnist"

    vertex_config = VertexConfig.from_env()
    if not vertex_config:
        raise SystemExit("GOOGLE_CLOUD_PROJECT and VERTEX_STAGING_BUCKET required")

    bucket = vertex_config.staging_bucket
    base_uri = f"gs://{bucket}/models"

    if case == "lora":
        # LoRA adapters are a directory
        if checkpoint_path.is_dir():
            upload_directory(checkpoint_path, f"{base_uri}/{run_id}/")
            upload_directory(checkpoint_path, f"{base_uri}/lora/latest/")
        else:
            upload_model(checkpoint_path, f"{base_uri}/{run_id}/adapter_model.safetensors")
            upload_model(checkpoint_path, f"{base_uri}/lora/latest/adapter_model.safetensors")
        register_model(f"{base_uri}/{run_id}/", f"lora-{run_id}", vertex_config)
    else:
        # MNIST: single file
        upload_model(checkpoint_path, f"{base_uri}/{run_id}/model.pt")
        upload_model(checkpoint_path, f"{base_uri}/latest/model.pt")
        register_model(f"{base_uri}/{run_id}/model.pt", f"mnist-{run_id}", vertex_config)

    end_run()


if __name__ == "__main__":
    main()
