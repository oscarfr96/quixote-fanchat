"""Una base SQLite para los runs, las métricas y los spans de MLflow."""

import mlflow
from mlflow import MlflowClient

from quixote.config import Settings


def configure_mlflow(settings: Settings) -> str:
    settings.runtime.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.experiment)
    if experiment is None:
        artifact_path = settings.runtime / "artifacts"
        artifact_path.mkdir(exist_ok=True)
        experiment_id = client.create_experiment(
            settings.experiment, artifact_location=artifact_path.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_usage(span, usage, model: str) -> dict[str, int]:
    tokens = {
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
    }
    # Nombres estándar: la interfaz nativa de MLflow también muestra los tokens.
    span.set_attributes({
        "mlflow.llm.model": model,
        "mlflow.chat.tokenUsage": {**tokens, "total_tokens": sum(tokens.values())},
    })
    return tokens


def recent_runs(settings: Settings, experiment_id: str, limit: int = 100):
    return MlflowClient(tracking_uri=settings.tracking_uri).search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.kind = 'question'",
        order_by=["attributes.start_time DESC"],
        max_results=limit,
    )
