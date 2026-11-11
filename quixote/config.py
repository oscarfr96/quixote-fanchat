"""Configuración local. La clave nunca se imprime ni se envía a MLflow."""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
import os

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    api_key: str = field(default="", repr=False)
    model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    experiment: str = "quixote-fanchat"

    @property
    def pdf(self) -> Path:
        return self.root / "assets" / "quijote.pdf"

    @property
    def runtime(self) -> Path:
        return self.root / ".runtime"

    @property
    def tracking_uri(self) -> str:
        return "sqlite:///" + (self.runtime / "mlflow.db").as_posix()


def load_settings() -> Settings:
    # Leer un archivo concreto evita cargar por accidente un .env de otro repo.
    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    return Settings(
        api_key=(values.get("OPENAI_API_KEY") or "").strip(),
        model=(values.get("OPENAI_API_MODEL") or "gpt-6-luna").strip(),
        embedding_model=(values.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip(),
    )
