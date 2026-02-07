from pathlib import Path

from google.cloud import aiplatform, storage

from src.config import VertexConfig


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


def register_model(gcs_uri: str, display_name: str, config: VertexConfig) -> None:
    aiplatform.init(
        project=config.project_id,
        location=config.location,
        staging_bucket=config.staging_bucket,
    )
    # Vertex AI expects artifact_uri to be a directory containing the model files
    artifact_dir = gcs_uri.rsplit("/", 1)[0] + "/"
    serving_image = (
        f"{config.location}-docker.pkg.dev/{config.project_id}/distributed-training/serve:latest"
    )
    aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_dir,
        serving_container_image_uri=serving_image,
    )
