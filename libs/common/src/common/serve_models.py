"""Base Pydantic models for predict request/response — no dict I/O."""

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    """Base for predict request — cases subclass with their fields."""

    model_config = ConfigDict(extra="forbid")


class PredictResponse(BaseModel):
    """Base for predict response — cases subclass with their fields."""

    model_config = ConfigDict(extra="forbid")


class PredictRequestBody(BaseModel):
    """Accepts any JSON at route boundary; case validates via RequestModel."""

    model_config = ConfigDict(extra="allow")
