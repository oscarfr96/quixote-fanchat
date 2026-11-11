"""Las tres pantallas deben renderizar sin depender de llamadas a OpenAI."""

from streamlit.testing.v1 import AppTest

from quixote.config import ROOT, Settings
import quixote.ui as ui
from quixote.telemetry import configure_mlflow


def test_chat_renders_without_api_key(monkeypatch):
    monkeypatch.setattr(ui, "load_settings", lambda: Settings(api_key=""))
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("OPENAI_API_KEY" in info.value for info in app.info)


def test_learning_page_renders_real_code_and_three_diagrams():
    app = AppTest.from_string("from quixote.ui import how_it_works_page\nhow_it_works_page()\n").run()
    assert not app.exception
    assert len(app.tabs) == 3
    assert any("self.client.responses.parse" in code.value for code in app.code)
    assert any("index.vectors @ query_vector" in code.value for code in app.code)
    assert any("mlflow.log_metrics(metrics)" in code.value for code in app.code)


def test_empty_observability_gives_a_useful_next_step(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path)
    experiment_id = configure_mlflow(settings)
    monkeypatch.setattr(ui, "telemetry", lambda: (settings, experiment_id))
    # Los SVG se leen desde ROOT, no desde la carpeta de datos de prueba.
    app = AppTest.from_string("from quixote.ui import observability_page\nobservability_page()\n").run()
    assert not app.exception
    assert any("todavía no tiene huellas" in info.value for info in app.info)
