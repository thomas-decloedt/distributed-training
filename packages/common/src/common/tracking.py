from google.cloud import aiplatform

from common.config import VertexConfig


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
    """End the current Vertex AI experiment run.

    No-op if no run is active (e.g. in upload_standalone subprocess).
    """
    try:
        aiplatform.end_run()
    except ValueError:
        pass  # No experiment/run in this process (e.g. upload_standalone)
