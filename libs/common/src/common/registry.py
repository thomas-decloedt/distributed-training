from pathlib import Path

from google.cloud import aiplatform, storage

from common.config import VertexConfig


def upload_model(checkpoint_path: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else str(checkpoint_path.name)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(checkpoint_path))


def upload_directory(local_dir: Path, gcs_uri: str) -> None:
    """Upload a directory to GCS (for LoRA adapters)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for f in local_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir)
            blob_path = f"{prefix.rstrip('/')}/{rel}" if prefix else str(rel)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(f))


def register_model(gcs_uri: str, display_name: str, config: VertexConfig) -> None:
    aiplatform.init(
        project=config.project_id,
        location=config.location,
        staging_bucket=config.staging_bucket,
    )
    artifact_dir = gcs_uri.rsplit("/", 1)[0] + "/"
    serving_image = (
        f"{config.location}-docker.pkg.dev/{config.project_id}/distributed-training/serve:latest"
    )
    aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_dir,
        serving_container_image_uri=serving_image,
    )
