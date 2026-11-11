"""Pruebas de comportamiento sin conexión ni gasto de API."""

from dataclasses import replace
from types import SimpleNamespace
import json

from mlflow import MlflowClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
import mlflow
import numpy as np
import pytest

from quixote.bot import GroundedAnswer, QuixoteBot, TopicDecision
from quixote.config import Settings
from quixote.prompts import NO_EVIDENCE, REFUSAL
from quixote.provider import ProviderError
from quixote.rag import BookIndex, Passage, load_or_build_index
from quixote.telemetry import configure_mlflow


class FakeResponses:
    def __init__(self, accepted=True, pages=None, failure=False):
        self.accepted = accepted
        self.pages = pages if pages is not None else [1]
        self.failure = failure
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise ProviderError("Proveedor no disponible")
        if kwargs["text_format"] is TopicDecision:
            parsed = TopicDecision(is_quijote=self.accepted, reason="Motivo de prueba",
                                   search_query="Sancho Panza escudero" if self.accepted else "")
        else:
            parsed = GroundedAnswer(supported=True, answer="Sancho sirve de escudero, vuestra merced.",
                                    pages=self.pages)
        return SimpleNamespace(output_parsed=parsed,
                               usage=SimpleNamespace(input_tokens=20, output_tokens=10))


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        texts = [kwargs["input"]] if isinstance(kwargs["input"], str) else kwargs["input"]
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0])
                                     for i in range(len(texts))],
                               usage=SimpleNamespace(total_tokens=5 * len(texts)))


def fake_client(**kwargs):
    return SimpleNamespace(responses=FakeResponses(**kwargs), embeddings=FakeEmbeddings())


def write_pdf(path, text="Sancho Panza era el escudero de don Quijote."):
    path.parent.mkdir(exist_ok=True, parents=True)
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as file:
        writer.write(file)


@pytest.fixture
def setup(tmp_path):
    settings = Settings(root=tmp_path, api_key="FAKE_SECRET_DO_NOT_LOG", top_k=1)
    experiment_id = configure_mlflow(settings)
    yield settings, experiment_id
    mlflow.flush_trace_async_logging()
    if mlflow.active_run():
        mlflow.end_run()


def seeded_bot(settings, experiment_id, client):
    instance = QuixoteBot(settings, experiment_id, client)
    instance.index = BookIndex([Passage("Sancho Panza era el escudero de don Quijote.", 1, 0)],
                               np.asarray([[1., 0.]], dtype=np.float32), 1, "fake-hash")
    return instance


def test_out_of_scope_never_reads_pdf_or_calls_embeddings(setup):
    settings, experiment_id = setup
    client = fake_client(accepted=False)
    reply = QuixoteBot(settings, experiment_id, client).answer("Como preparo una tortilla?")
    assert reply.text == REFUSAL
    assert reply.route == "rechazo"
    assert not reply.passages
    assert not client.embeddings.calls
    assert len(client.responses.calls) == 1
    run = MlflowClient().get_run(reply.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["total_tokens"] == 30
    assert run.data.metrics["retrieved_chunks"] == 0
    trace = MlflowClient().get_trace(reply.trace_id)
    assert trace.info.token_usage["total_tokens"] == 30
    assert {s.name for s in trace.data.spans} == {"pregunta_quijote", "clasificar_tema", "rechazar_con_cortesia"}
    assert "FAKE_SECRET_DO_NOT_LOG" not in trace.to_json()


def test_rag_persists_connected_spans_sources_usage_and_history(setup):
    settings, experiment_id = setup
    client = fake_client()
    history = [{"role": "user", "content": "Quien es don Quijote?"},
               {"role": "assistant", "content": "Un hidalgo."}]
    instance = seeded_bot(settings, experiment_id, client)
    progress = []
    reply = instance.answer("Y su escudero?", history, on_reading=lambda: progress.append(True))
    assert progress == [True]
    assert reply.route == "rag" and reply.cited_pages == [1]
    assert reply.passages[0].page == 1
    assert json.loads(client.responses.calls[0]["input"])["history"] == history
    assert len(client.responses.calls) == 2
    run = MlflowClient().get_run(reply.run_id)
    assert run.data.metrics["total_tokens"] == 60
    assert run.data.metrics["embedding_tokens"] == 5
    assert run.data.metrics["retrieved_chunks"] == 1
    trace = MlflowClient().get_trace(reply.trace_id)
    assert trace.info.token_usage["total_tokens"] == 60
    root = next(s for s in trace.data.spans if s.name == "pregunta_quijote")
    assert len(trace.data.spans) == 5
    assert all(s.trace_id == root.trace_id for s in trace.data.spans)
    assert all(s.parent_id is not None for s in trace.data.spans if s.name != "pregunta_quijote")


def test_hallucinated_page_falls_back_to_no_evidence(setup):
    settings, experiment_id = setup
    reply = seeded_bot(settings, experiment_id, fake_client(pages=[999])).answer("Quien es Sancho?")
    assert reply.text == NO_EVIDENCE
    assert reply.cited_pages == []
    assert reply.metrics["grounded_answer"] == 0


def test_api_failure_is_visible_as_failed_run_and_error_trace(setup):
    settings, experiment_id = setup
    with pytest.raises(ProviderError):
        QuixoteBot(settings, experiment_id, fake_client(failure=True)).answer("Quien es Sancho?")
    runs = MlflowClient().search_runs([experiment_id], filter_string="tags.kind = 'question'")
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
    assert runs[0].data.metrics["error"] == 1
    traces = mlflow.search_traces(locations=[experiment_id], return_type="list")
    assert len(traces) == 1
    assert "ERROR" in str(traces[0].info.state)


def test_index_survives_reload_and_invalidates_when_pdf_changes(setup):
    settings, experiment_id = setup
    write_pdf(settings.pdf)
    client = fake_client()
    first = load_or_build_index(settings, client, experiment_id)
    assert first.pages == 1 and "Sancho" in first.passages[0].text
    assert len(client.embeddings.calls) == 1
    second = load_or_build_index(settings, client, experiment_id)
    assert len(client.embeddings.calls) == 1  # La segunda carga no gasta embeddings.
    np.testing.assert_array_equal(first.vectors, second.vectors)
    write_pdf(settings.pdf, "Rocinante era el caballo de don Quijote.")
    third = load_or_build_index(settings, client, experiment_id)
    assert third.fingerprint != first.fingerprint
    assert len(client.embeddings.calls) == 2
    assert len(list((settings.runtime / "index").glob("*.npz"))) == 2


def test_different_embedding_model_creates_new_index(setup):
    settings, experiment_id = setup
    write_pdf(settings.pdf)
    client = fake_client()
    load_or_build_index(settings, client, experiment_id)
    load_or_build_index(replace(settings, embedding_model="other-model"), client, experiment_id)
    assert len(client.embeddings.calls) == 2


def test_empty_input_does_not_create_a_run(setup):
    settings, experiment_id = setup
    instance = QuixoteBot(settings, experiment_id, fake_client())
    with pytest.raises(ValueError):
        instance.answer("  ")
    assert not MlflowClient().search_runs([experiment_id])
