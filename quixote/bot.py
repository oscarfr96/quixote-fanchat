"""Clasificar -> rechazar o recuperar -> responder, dentro de una traza."""

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
import json

from openai import OpenAI
from pydantic import BaseModel, Field
import mlflow

from quixote.config import Settings
from quixote.prompts import ANSWER_PROMPT, CLASSIFIER_PROMPT, NO_EVIDENCE, REFUSAL
from quixote.provider import ProviderError, call_api
from quixote.rag import Passage, load_or_build_index, retrieve
from quixote.telemetry import log_usage


class TopicDecision(BaseModel):
    is_quijote: bool
    reason: str
    search_query: str


class GroundedAnswer(BaseModel):
    supported: bool
    answer: str
    pages: list[int] = Field(description="Páginas del PDF usadas como evidencia")


@dataclass
class Reply:
    text: str
    route: str
    reason: str
    passages: list[Passage] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    trace_id: str = ""


class QuixoteBot:
    def __init__(self, settings: Settings, experiment_id: str, client=None):
        if not settings.api_key and client is None:
            raise ValueError("Añade OPENAI_API_KEY a tu archivo .env para conversar.")
        self.settings = settings
        self.experiment_id = experiment_id
        self.client = client or OpenAI(api_key=settings.api_key, timeout=60, max_retries=1)
        self.index = None
        self._index_lock = Lock()

    def classify(self, question: str, history: list[dict]) -> tuple[TopicDecision, dict]:
        with mlflow.start_span(name="clasificar_tema", span_type="LLM") as span:
            span.set_inputs({"question": question, "history": history})
            response = call_api(self.client.responses.parse,
                model=self.settings.model,
                instructions=CLASSIFIER_PROMPT,
                input=json.dumps({"history": history, "question": question}, ensure_ascii=False),
                text_format=TopicDecision,
                store=False,
            )
            decision = response.output_parsed
            if decision is None:
                raise ProviderError("El modelo no pudo clasificar la pregunta. Prueba a reformularla.")
            span.set_outputs(decision.model_dump())
            return decision, log_usage(span, response.usage, self.settings.model)

    def generate(self, question: str, history: list[dict], passages: list[Passage]):
        with mlflow.start_span(name="redactar_respuesta", span_type="LLM") as span:
            context = [{"page": p.page, "text": p.text} for p in passages]
            span.set_inputs({"question": question, "context": context})
            response = call_api(self.client.responses.parse,
                model=self.settings.model,
                instructions=ANSWER_PROMPT,
                input=json.dumps({"history": history, "question": question,
                                  "pdf_fragments": context}, ensure_ascii=False),
                text_format=GroundedAnswer,
                store=False,
            )
            draft = response.output_parsed
            if draft is None:
                raise ProviderError("El modelo no pudo redactar la respuesta. Prueba otra pregunta sobre el libro.")
            # Rechazar citas ajenas a los fragmentos recuperados.
            available_pages = {p.page for p in passages}
            valid = (draft.supported and bool(draft.pages) and bool(draft.answer.strip())
                     and set(draft.pages).issubset(available_pages))
            text = draft.answer if valid else NO_EVIDENCE
            cited_pages = sorted(set(draft.pages)) if valid else []
            span.set_outputs({"answer": text, "pages": cited_pages, "supported": bool(valid)})
            return text, cited_pages, log_usage(span, response.usage, self.settings.model)

    def answer(self, question: str, history: list[dict] | None = None,
               session_id: str = "", on_reading=None) -> Reply:
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError("Escribe una pregunta de entre 1 y 2000 caracteres.")
        # Solo se conserva una ventana breve para resolver las repreguntas.
        history = [{"role": m["role"], "content": str(m["content"])[:2500]}
                   for m in (history or [])[-6:] if m["role"] in {"user", "assistant"}]
        started = perf_counter()
        metrics = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0,
                   "retrieved_chunks": 0, "classification_ms": 0,
                   "retrieval_ms": 0, "generation_ms": 0, "error": 0}
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=question[:70],
                              tags={"kind": "question", "session_id": session_id}) as run:
            mlflow.log_params({"model": self.settings.model,
                               "embedding_model": self.settings.embedding_model,
                               "top_k": self.settings.top_k,
                               "chunk_size": self.settings.chunk_size,
                               "chunk_overlap": self.settings.chunk_overlap})
            try:
                # docs:start trace
                with mlflow.start_span(name="pregunta_quijote", span_type="CHAIN",
                                       run_id=run.info.run_id) as root_span:
                    root_span.set_inputs({"question": question})
                    mlflow.update_current_trace(session_id=session_id or None, request_preview=question)
                    trace_id = root_span.trace_id
                    mlflow.set_tag("trace_id", trace_id)
                    stage = perf_counter()
                    decision, usage = self.classify(question, history)
                    metrics["classification_ms"] = (perf_counter() - stage) * 1000
                    metrics.update(usage)
                    route = "rag" if decision.is_quijote else "rechazo"
                    mlflow.set_tags({"route": route, "reason": decision.reason[:1000]})
                    mlflow.update_current_trace(tags={"route": route})
                    passages, cited_pages = [], []
                    if not decision.is_quijote:
                        with mlflow.start_span(name="rechazar_con_cortesia", span_type="CHAIN") as refusal_span:
                            text = REFUSAL
                            refusal_span.set_outputs({"answer": text})
                    else:
                        if on_reading:
                            on_reading()
                        stage = perf_counter()
                        with self._index_lock:
                            if self.index is None:
                                self.index = load_or_build_index(self.settings, self.client, self.experiment_id)
                        metrics["index_ready_ms"] = (perf_counter() - stage) * 1000
                        mlflow.log_param("pdf_sha256", self.index.fingerprint)
                        stage = perf_counter()
                        passages, embedding_tokens = retrieve(
                            self.index, decision.search_query or question, self.client, self.settings)
                        metrics["retrieval_ms"] = (perf_counter() - stage) * 1000
                        metrics["embedding_tokens"] = embedding_tokens
                        metrics["retrieved_chunks"] = len(passages)
                        stage = perf_counter()
                        text, cited_pages, usage = self.generate(question, history, passages)
                        metrics["generation_ms"] = (perf_counter() - stage) * 1000
                        metrics["input_tokens"] += usage["input_tokens"]
                        metrics["output_tokens"] += usage["output_tokens"]
                    metrics["grounded_answer"] = int(bool(cited_pages))
                    root_span.set_outputs({"answer": text, "route": route, "pages": cited_pages})
                    mlflow.update_current_trace(response_preview=text)
                    root_span.set_attributes({"route": route, "classification_reason": decision.reason})
                # docs:end trace
                return Reply(text, route, decision.reason, passages, cited_pages,
                             metrics, run.info.run_id, trace_id)
            except Exception as exc:
                metrics["error"] = 1
                mlflow.set_tag("error_type", type(exc).__name__)
                raise
            finally:
                # docs:start metrics
                metrics["latency_ms"] = (perf_counter() - started) * 1000
                metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
                mlflow.log_metrics(metrics)
                # docs:end metrics
                mlflow.flush_trace_async_logging()
