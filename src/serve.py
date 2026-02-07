import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import MnistCNN

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "./checkpoints/model.pt")
MEAN = 0.1307
STD = 0.3081

_model: MnistCNN | None = None


def load_model() -> MnistCNN:
    global _model
    if _model is not None:
        return _model
    path = MODEL_PATH
    if path.startswith("gs://"):
        from google.cloud import storage

        parts = path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1]
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        local_path = Path("/tmp/model.pt")
        blob.download_to_filename(str(local_path))
        path = str(local_path)
    model = MnistCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    _model = model
    return _model


class PredictRequest(BaseModel):
    image: list[list[float]]


class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = load_model()
    tensor = torch.tensor([req.image], dtype=torch.float32)
    tensor = tensor.view(-1, 1, 28, 28)
    tensor = (tensor - MEAN) / STD
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].tolist()
    pred = int(torch.argmax(logits, dim=1)[0].item())
    return PredictResponse(prediction=pred, probabilities=probs)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
