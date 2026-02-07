from google.cloud import aiplatform

from src.config import VertexConfig


def init_experiment(config: VertexConfig) -> None:
    aiplatform.init(
        project=config.project_id,
        location=config.location,
        experiment=config.experiment_name,
    )


def start_run(run_name: str) -> None:
    aiplatform.start_run(run_name)


def log_metrics(metrics: dict[str, float]) -> None:
    aiplatform.log_metrics(metrics)


def log_params(params: dict[str, str | int | float]) -> None:
    aiplatform.log_params(params)


def end_run() -> None:
    aiplatform.end_run()
