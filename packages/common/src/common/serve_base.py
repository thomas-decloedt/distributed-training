"""CaseProtocol: interface for case-specific model loading and prediction."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from common.serve_models import PredictRequest, PredictResponse

T = TypeVar("T")


class CaseProtocol(Protocol[T]):
    """Interface each case (mnist, lora) implements for the reusable serve app."""

    RequestModel: type[PredictRequest]
    ResponseModel: type[PredictResponse]

    def load(self, path: str) -> T:
        """Load model from path (local or gs://). Return the loaded model."""
        ...

    def predict(self, model: T, request: BaseModel) -> BaseModel:
        """Run prediction. Returns Pydantic response model."""
        ...


def download_from_gcs(gcs_path: str, local_path: str) -> str:
    """Download a file from GCS. Returns local path."""
    from pathlib import Path

    from google.cloud import storage

    if not gcs_path.startswith("gs://"):
        return gcs_path
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local))
    return str(local)
