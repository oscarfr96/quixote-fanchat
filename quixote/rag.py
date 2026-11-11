"""RAG pequeño: PDF -> fragmentos -> embeddings -> similitud del coseno."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import mlflow
import numpy as np

from quixote.config import Settings
from quixote.provider import call_api


@dataclass(frozen=True)
class Passage:
    text: str
    page: int
    chunk_id: int
    score: float = 0.0


@dataclass
class BookIndex:
    passages: list[Passage]
    vectors: np.ndarray
    pages: int
    fingerprint: str


def split_pdf(settings: Settings) -> tuple[list[Passage], int]:
    reader = PdfReader(settings.pdf)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    passages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Mantener la página original permite enseñar de dónde salió la respuesta.
        for chunk in splitter.split_text(text):
            if chunk.strip():
                passages.append(Passage(chunk, page_number, len(passages)))
    if not passages:
        raise ValueError("El PDF no contiene texto extraíble. Esta demo no incluye OCR.")
    return passages, len(reader.pages)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def load_or_build_index(settings: Settings, client, experiment_id: str) -> BookIndex:
    if not settings.pdf.is_file():
        raise FileNotFoundError("Coloca el libro en assets/quijote.pdf.")
    fingerprint = sha256(settings.pdf.read_bytes()).hexdigest()
    cache_key = sha256(json.dumps([
        fingerprint, settings.embedding_model, settings.chunk_size,
        settings.chunk_overlap, "pypdf-v1",
    ]).encode()).hexdigest()[:24]
    cache_dir = settings.runtime / "index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.npz"
    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                passages = [Passage(str(text), int(page), i) for i, (text, page)
                            in enumerate(zip(data["texts"], data["page_numbers"], strict=True))]
                vectors = data["vectors"]
                if vectors.ndim != 2 or len(vectors) != len(passages) or not len(passages):
                    raise ValueError("Índice incompleto")
                return BookIndex(passages, vectors, int(data["pages"]), fingerprint)
        except (OSError, ValueError, KeyError):
            # Una caché incompleta se reconstruye; nunca se deserializa pickle.
            pass

    started = perf_counter()
    with mlflow.start_run(experiment_id=experiment_id, run_name="Preparar el libro",
                          nested=True, tags={"kind": "index"}):
        with mlflow.start_span(name="indexar_pdf", span_type="CHAIN") as span:
            passages, pages = split_pdf(settings)
            vectors, token_count = [], 0
            # Lotes pequeños; cada fragmento tiene como máximo 1200 caracteres.
            for offset in range(0, len(passages), 96):
                batch = passages[offset:offset + 96]
                response = call_api(client.embeddings.create,
                    model=settings.embedding_model, input=[p.text for p in batch])
                vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
                token_count += response.usage.total_tokens
            matrix = normalize(np.asarray(vectors, dtype=np.float32))
            # Un solo archivo y reemplazo atómico: un cierre no deja dos archivos desparejados.
            temporary = cache_dir / f"{cache_key}-{uuid4().hex}.tmp"
            try:
                with temporary.open("wb") as file:
                    np.savez_compressed(file, vectors=matrix,
                        texts=np.asarray([p.text for p in passages]),
                        page_numbers=np.asarray([p.page for p in passages]), pages=pages)
                temporary.replace(cache_file)
            finally:
                temporary.unlink(missing_ok=True)
            mlflow.log_params({"embedding_model": settings.embedding_model,
                               "chunk_size": settings.chunk_size,
                               "chunk_overlap": settings.chunk_overlap, "pdf_sha256": fingerprint})
            mlflow.log_metrics({"pages": pages, "chunks": len(passages),
                                "embedding_tokens": token_count,
                                "latency_ms": (perf_counter() - started) * 1000})
            span.set_outputs({"pages": pages, "chunks": len(passages), "embedding_tokens": token_count})
    return BookIndex(passages, matrix, pages, fingerprint)


def retrieve(index: BookIndex, query: str, client, settings: Settings) -> tuple[list[Passage], int]:
    with mlflow.start_span(name="recuperar_fragmentos", span_type="RETRIEVER") as span:
        span.set_inputs({"query": query, "top_k": settings.top_k})
        with mlflow.start_span(name="embedding_pregunta", span_type="EMBEDDING") as embedding_span:
            response = call_api(client.embeddings.create,
                model=settings.embedding_model, input=query)
            query_vector = normalize(np.asarray(response.data[0].embedding, dtype=np.float32))
            embedding_span.set_attributes({"model": settings.embedding_model,
                                            "embedding_tokens": response.usage.total_tokens})
        # Vectores normalizados: el producto escalar es la similitud del coseno.
        scores = index.vectors @ query_vector
        positions = np.argsort(scores)[-settings.top_k:][::-1]
        found = [Passage(index.passages[i].text, index.passages[i].page,
                         index.passages[i].chunk_id, float(scores[i])) for i in positions]
        span.set_outputs([{"page_content": p.text,
                           "metadata": {"page": p.page, "chunk_id": p.chunk_id, "score": p.score}}
                          for p in found])
        return found, response.usage.total_tokens
