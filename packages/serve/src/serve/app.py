"""Reusable FastAPI app that delegates to case (mnist, lora) via MODEL_CASE env."""

import os

from common.serve_base import CaseProtocol
from common.serve_models import PredictRequestBody, PredictResponse
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_CASE = os.environ.get("MODEL_CASE", "mnist")
MODEL_PATH = os.environ.get("MODEL_PATH", "./checkpoints/model.pt")

_case: CaseProtocol | None = None
_model: object | None = None


def _get_case() -> CaseProtocol:
    global _case
    if _case is None:
        _case = _resolve_case(MODEL_CASE)
    return _case


def _resolve_case(name: str) -> CaseProtocol:
    """Load the case module by name. Raises ValueError if unknown."""
    if name == "mnist":
        from mnist.serve import MnistCase

        return MnistCase()
    if name == "lora":
        from lora.serve import LoraCase

        return LoraCase()
    raise ValueError(f"Unknown case: {name}")


def _load_model() -> object:
    global _model
    if _model is not None:
        return _model
    case = _get_case()
    _model = case.load(MODEL_PATH)
    return _model


_case_for_route = _get_case()


@app.post("/predict", response_model=_case_for_route.ResponseModel)
def predict(request: PredictRequestBody) -> PredictResponse:
    """Predict: FastAPI parses body, case validates via RequestModel."""
    req = _case_for_route.RequestModel.model_validate(request.model_dump())
    model = _load_model()
    return _case_for_route.predict(model, req)


class HealthResponse(BaseModel):
    status: str
    case: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", case=MODEL_CASE)
