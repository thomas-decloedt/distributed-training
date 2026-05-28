"""MnistCase implements CaseProtocol for the reusable serve app."""

import torch
from common.serve_base import download_from_gcs
from common.serve_models import PredictRequest, PredictResponse

from mnist.model import MnistCNN

MEAN = 0.1307
STD = 0.3081


class MnistPredictRequest(PredictRequest):
    """Request: flattened 28x28 image as list of floats."""

    image: list[float] = []


class MnistPredictResponse(PredictResponse):
    """Response: predicted class and class probabilities."""

    prediction: int
    probabilities: list[float]


class MnistCase:
    RequestModel = MnistPredictRequest
    ResponseModel = MnistPredictResponse

    def load(self, path: str) -> MnistCNN:
        local = path
        if path.startswith("gs://"):
            local = download_from_gcs(path, "/tmp/model.pt")
        model = MnistCNN()
        model.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
        model.eval()
        return model

    def predict(self, model: MnistCNN, request: MnistPredictRequest) -> MnistPredictResponse:
        tensor = torch.tensor([request.image], dtype=torch.float32)
        tensor = tensor.view(-1, 1, 28, 28)
        tensor = (tensor - MEAN) / STD
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        pred = int(torch.argmax(logits, dim=1)[0].item())
        return MnistPredictResponse(prediction=pred, probabilities=probs)
