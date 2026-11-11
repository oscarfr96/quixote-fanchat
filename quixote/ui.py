"""Tres páginas. La interfaz explica el pipeline con su propio código fuente."""

# En Streamlit, al interactuar con la página, normalmente vuelve a ejecutar el código de Python de
# arriba a abajo. Por eso necesitamos conservar la conversación entre ejecuciones.
# Eso lo hacemos con st.session_state.setdefault("messages", [])
# Cada mensaje lo guarda con el formato: {"role": "user", "content": "¿Quién es Sancho Panza?"}

from datetime import datetime
from uuid import uuid4
import ast
import textwrap

from mlflow import MlflowClient
import pandas as pd
import streamlit as st

from quixote.bot import QuixoteBot, Reply
from quixote.config import ROOT, load_settings
from quixote.prompts import READING
from quixote.telemetry import configure_mlflow, recent_runs


def apply_style():
    st.markdown("""<style>
    .stApp { background: #fff; }
    .block-container { max-width: 960px; padding-top: 3.5rem; padding-bottom: 1.5rem; }
    h1 { font-size: 1.8rem !important; font-weight: 600 !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.1rem !important; }
    .eyebrow { color: #777; font-size: .75rem; margin-bottom: .5rem; }
    [data-testid="stPageLink"] { font-size: .85rem; }
    </style>""", unsafe_allow_html=True)


@st.cache_resource
def telemetry():
    settings = load_settings()
    return settings, configure_mlflow(settings)


@st.cache_resource
def bot():
    settings, experiment_id = telemetry()
    return QuixoteBot(settings, experiment_id)


def native_url(experiment_id: str, run_id: str = "") -> str:
    base = f"http://127.0.0.1:5000/#/experiments/{experiment_id}"
    return f"{base}/runs/{run_id}" if run_id else base


def source_details(reply: Reply):
    if reply.cited_pages:
        st.caption("Páginas del PDF: " + ", ".join(str(page) for page in reply.cited_pages))
    if reply.passages:
        with st.expander("Fuentes"):
            for passage in reply.passages:
                used = " · citada en la respuesta" if passage.page in reply.cited_pages else ""
                st.markdown(f"**Página {passage.page}{used}**")
                st.text(passage.text)


def chat_page():
    settings = load_settings()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid4().hex)
    heading, clear = st.columns([4, 1.5], vertical_alignment="center")
    with heading:
        st.title("El Quijote")
    with clear:
        if st.button("Nueva charla", icon=":material/restart_alt:"):
            st.session_state.messages = []
            st.session_state.session_id = uuid4().hex
            st.rerun()
    st.caption("Preguntas y respuestas sobre la novela de Cervantes.")
    if not settings.api_key:
        st.info("Para empezar, copia .env.example como .env y añade tu OPENAI_API_KEY. El recorrido «Cómo funciona» está disponible sin clave.")
    if not st.session_state.messages:
        st.caption("Prueba con «¿Quién es Sancho Panza?».")
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🪶" if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            if "reply" in message:
                source_details(message["reply"])
    question = st.chat_input("Escribid vuestra pregunta sobre El Quijote…", max_chars=2000,
                             disabled=not settings.api_key)
    if question and settings.api_key:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🪶"):
            progress = st.empty()
            progress.caption("Un instante, vuestra merced…")
            def reading():
                progress.caption(READING)
            try:
                reply = bot().answer(question, history, st.session_state.session_id, reading)
            except Exception as exc:
                progress.empty()
                from quixote.provider import ProviderError
                if isinstance(exc, (ProviderError, ValueError, FileNotFoundError)):
                    error = str(exc)
                else:
                    error = ("La consulta no pudo completarse. Revisa el run fallido en Observabilidad "
                             f"y la consola local ({type(exc).__name__}).")
                st.error(error)
                # No introducir un error técnico como respuesta literaria en el historial.
                st.session_state.messages.pop()
            else:
                progress.empty()
                st.markdown(reply.text)
                source_details(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply.text, "reply": reply})


def observability_page():
    st.markdown('<div class="eyebrow">OBSERVABILIDAD · MLFLOW EN VIVO</div>', unsafe_allow_html=True)
    st.title("Cada pregunta deja huella.")
    st.write("Mira qué decidió el modelo, cuánto tardó cada paso y qué páginas leyó. Estos datos vienen de MLflow, no de una simulación.")
    settings, experiment_id = telemetry()
    left, middle, right = st.columns([3, 2, 1])
    with left:
        st.link_button("Abrir la plataforma MLflow ↗", native_url(experiment_id), type="primary")
    with middle:
        limit = st.selectbox("Preguntas recientes", [25, 100, 500], index=1)
    with right:
        st.button("Actualizar", icon=":material/refresh:")
    st.caption("El botón abre MLflow en el puerto 5000. Se inicia junto al chat con `python run.py`. Dentro del experimento, abre Traces para ver el árbol de llamadas.")
    runs = recent_runs(settings, experiment_id, limit)
    if not runs:
        st.info("La biblioteca todavía no tiene huellas. Haz una pregunta sobre el libro y otra fuera de tema; vuelve aquí y pulsa Actualizar.")
        st.image(str(ROOT / "assets" / "tracing.svg"), width="stretch")
        return
    rows = []
    for run in runs:
        metrics, tags = run.data.metrics, run.data.tags
        rows.append({"Hora": datetime.fromtimestamp(run.info.start_time / 1000).astimezone(),
                     "Pregunta": tags.get("mlflow.runName", ""),
                     "Ruta": tags.get("route", "sin completar"), "Estado": run.info.status,
                     "Segundos": metrics.get("latency_ms", 0) / 1000,
                     "Tokens LLM": int(metrics.get("total_tokens", 0)),
                     "Tokens embeddings": int(metrics.get("embedding_tokens", 0)),
                     "Fragmentos": int(metrics.get("retrieved_chunks", 0)), "run_id": run.info.run_id})
    frame = pd.DataFrame(rows)
    complete = frame[frame["Estado"] == "FINISHED"]
    cols = st.columns(4)
    cols[0].metric("Preguntas", len(frame))
    cols[1].metric("Latencia media", f"{complete['Segundos'].mean():.2f} s" if len(complete) else "—")
    cols[2].metric("Tokens del LLM", f"{frame['Tokens LLM'].sum():,}")
    cols[3].metric("Fuera de tema", int((frame["Ruta"] == "rechazo").sum()))
    st.caption(f"Ventana: últimas {len(frame)} preguntas. Fallidas: {(frame['Estado'] == 'FAILED').sum()}. "
               "Tokens de embeddings e indexación se registran aparte. La calidad no se evalúa automáticamente.")
    chart, table, index_tab = st.tabs(["Tiempos", "Preguntas y métricas", "Preparación del libro"])
    with chart:
        st.bar_chart(frame.set_index("Hora")[["Segundos"]].sort_index(), color="#7f8c6a")
    with table:
        st.dataframe(frame.drop(columns=["run_id"]), hide_index=True, width="stretch")
    with index_tab:
        index_runs = MlflowClient().search_runs([experiment_id], filter_string="tags.kind = 'index'",
                                               order_by=["attributes.start_time DESC"], max_results=10)
        st.write("El índice del PDF se reutiliza en las siguientes consultas, incluso si reinicias. Cambiar el PDF, el modelo de embeddings o el tamaño de fragmento crea otro índice.")
        if index_runs:
            st.dataframe(pd.DataFrame([{"Estado": r.info.status, **r.data.metrics} for r in index_runs]),
                         hide_index=True, width="stretch")
        else:
            st.info("Todavía no se ha preparado el libro en este experimento.")
    st.subheader("Seguid el hilo de una pregunta")
    by_id = {r.info.run_id: r for r in runs}
    run_id = st.selectbox("Elige una pregunta", list(by_id),
        format_func=lambda key: f"{by_id[key].data.tags.get('mlflow.runName', key)} · {by_id[key].info.status}")
    run = by_id[run_id]
    st.write("**Decisión del clasificador:** " + run.data.tags.get("reason", "La ejecución falló antes de la decisión."))
    st.link_button("Ver run y parámetros en MLflow ↗", native_url(experiment_id, run_id))
    trace_id = run.data.tags.get("trace_id")
    if trace_id:
        trace = MlflowClient().get_trace(trace_id)
        if trace is not None:
            spans = sorted(trace.data.spans, key=lambda span: span.start_time_ns)
            st.dataframe(pd.DataFrame([{"Paso": span.name, "Tipo": span.span_type,
                "Duración (ms)": round((span.end_time_ns - span.start_time_ns) / 1e6, 1),
                "Estado": str(span.status.status_code)} for span in spans]), hide_index=True, width="stretch")
            with st.expander("Entradas y salidas reales de cada paso"):
                for span in spans:
                    st.markdown(f"**{span.name}**")
                    st.json({"entrada": span.inputs, "salida": span.outputs}, expanded=False)
    else:
        st.info("Esta ejecución no tiene todavía una traza completa. Los errores se consultan también en MLflow → Traces.")
    with st.expander("Qué significa cada dato"):
        st.markdown("""
        - **Run:** una ejecución; aquí, una pregunta. Agrupa parámetros, métricas y etiquetas.
        - **Trace:** el recorrido completo de la pregunta.
        - **Span:** un paso de ese recorrido, con sus entradas, salidas, duración y estado.
        - **Parámetros:** configuración: modelo, tamaño de fragmentos y `top_k`.
        - **Métricas:** números: tiempo, tokens, fragmentos y errores.
        - **Etiquetas:** contexto: ruta elegida, motivo y sesión.

        `grounded_answer=1` solo indica que el modelo declaró evidencia y citó páginas recuperadas;
        no verifica automáticamente que su respuesta sea verdadera.

        La plataforma nativa de MLflow puede mostrar un coste estimado a partir del modelo y sus tokens.
        Esa estimación es orientativa; no sustituye la facturación de OpenAI.
        """)


def show_source(relative_path: str, symbol: str | None = None, section: str | None = None):
    path = ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    start_line = 1
    if symbol:
        nodes = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol]
        node = nodes[0]
        start_line = node.lineno
        source = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    if section:
        lines = source.splitlines()
        start = next(i for i, line in enumerate(lines) if f"# docs:start {section}" in line)
        end = next(i for i, line in enumerate(lines) if f"# docs:end {section}" in line)
        start_line = start + 2
        source = "\n".join(lines[start + 1:end])
    st.caption(f"{relative_path} · línea {start_line} · leído del archivo real")
    st.code(textwrap.dedent(source), language="python", line_numbers=True)


def how_it_works_page():
    st.markdown('<div class="eyebrow">CÓMO FUNCIONA · DEL LIBRO A LA TRAZA</div>', unsafe_allow_html=True)
    st.title("Tres ideas. Una conversación.")
    st.write("Un recorrido breve para entender el proyecto y poder explicarlo en una entrevista. Despliega el código: es el mismo que ejecuta este chat.")
    classification, rag, mlflow_tab = st.tabs(["01 · ¿Es del Quijote?", "02 · Leer antes de responder", "03 · Observar con MLflow"])
    with classification:
        st.header("El modelo clasifica. Python elige el camino.")
        st.write("La primera llamada al LLM recibe instrucciones y una pregunta. Devuelve un objeto con un booleano, un motivo y una consulta de búsqueda. Pydantic define la forma de esa respuesta; el `if` de Python decide si hay que consultar el PDF.")
        st.image(str(ROOT / "assets" / "routing.svg"), width="stretch")
        st.json({"is_quijote": True, "reason": "Pregunta por un personaje de la novela.",
                 "search_query": "Sancho Panza escudero de don Quijote"}, expanded=True)
        st.caption("Ejemplo ilustrativo del formato; las decisiones reales están en Observabilidad.")
        st.write("No es un filtro de palabras: «¿Quién es Sancho?» puede pasar sin decir Quijote. «Explícame Python como si fueras Sancho» debe rechazarse porque su intención es ajena al libro. El historial permite entender «¿y su caballo?».")
        with st.expander("Ver el esquema de salida"):
            show_source("quixote/bot.py", "TopicDecision")
        with st.expander("Ver la clasificación real", expanded=True):
            show_source("quixote/bot.py", "classify")
        with st.expander("Ver las instrucciones del clasificador"):
            from quixote.prompts import CLASSIFIER_PROMPT
            st.code(CLASSIFIER_PROMPT, language="text")
        st.info("Un esquema garantiza el formato, no una decisión infalible. Esta demo usa instrucciones y una respuesta de rechazo fija; un producto público requeriría evaluar más casos y ataques.")
    with rag:
        st.header("El libro es la fuente; el LLM pone las palabras.")
        st.write("RAG significa Retrieval-Augmented Generation: buscar fragmentos relevantes y entregárselos al modelo junto con la pregunta. No entrenamos ni modificamos el LLM.")
        st.image(str(ROOT / "assets" / "rag.svg"), width="stretch")
        st.markdown("""
        1. **Leer y dividir:** `pypdf` extrae cada página; un splitter de LangChain crea fragmentos de 1200 caracteres con 200 de solapamiento. Cada fragmento conserva su página.
        2. **Representar:** OpenAI convierte los fragmentos en embeddings. Son listas de números que representan su significado. El índice se guarda localmente y se reutiliza.
        3. **Recuperar:** convertimos la pregunta en un vector, calculamos similitud del coseno y elegimos los 5 fragmentos más próximos.
        4. **Generar:** el LLM recibe pregunta + fragmentos. Responde en castellano de sabor áureo y señala las páginas utilizadas; si no encuentra evidencia, lo reconoce.
        """)
        st.write("Por ejemplo, una pregunta sobre los molinos se busca como un vector. El modelo de chat ve solo los fragmentos recuperados, no las 471 páginas en cada turno. La similitud no equivale a certeza y los fragmentos pueden no ser suficientes.")
        with st.expander("1 · Extraer y dividir el PDF", expanded=True):
            show_source("quixote/rag.py", "split_pdf")
        with st.expander("2 · Construir y guardar el índice"):
            show_source("quixote/rag.py", "load_or_build_index")
        with st.expander("3 · Recuperar por similitud"):
            show_source("quixote/rag.py", "retrieve")
        with st.expander("4 · Responder usando los fragmentos"):
            show_source("quixote/bot.py", "generate")
        st.info("Conservamos la extracción original en las fuentes. Los números mostrados son páginas del PDF, no números de capítulo. El modelo no recibe el libro entero en cada consulta.")
    with mlflow_tab:
        st.header("MLflow toma nota de lo que ocurre.")
        st.write("MLflow es una plataforma para registrar y comparar ejecuciones. En aplicaciones de IA generativa, sus trazas permiten inspeccionar llamadas al LLM y recuperaciones. No genera la respuesta: observa el pipeline.")
        st.image(str(ROOT / "assets" / "tracing.svg"), width="stretch")
        st.markdown("""
        En este proyecto, **un experimento** agrupa todas las pruebas, **un run** agrupa los datos de una pregunta,
        **una traza** cuenta su recorrido y **cada span** representa un paso. Un run adicional registra la creación del índice.

        - `mlflow.set_tracking_uri(...)` conecta con la base SQLite local.
        - `mlflow.start_run(...)` abre la ejecución de una pregunta.
        - `mlflow.start_span(...)` mide cada paso y relaciona sus hijos automáticamente.
        - `span.set_inputs(...)` y `span.set_outputs(...)` guardan qué entró y qué salió.
        - `mlflow.log_params(...)`, `mlflow.log_metrics(...)` y `mlflow.set_tags(...)` guardan configuración, números y contexto.
        """)
        with st.expander("Conectar MLflow", expanded=True):
            show_source("quixote/telemetry.py", "configure_mlflow")
        with st.expander("Crear la traza y sus pasos"):
            show_source("quixote/bot.py", section="trace")
        with st.expander("Guardar métricas al terminar, también si falla"):
            show_source("quixote/bot.py", section="metrics")
        st.subheader("Probadlo en dos minutos")
        st.markdown("""
        1. Pregunta **«¿Quién es Sancho Panza?»**. Verás clasificar → recuperar → redactar.
        2. Pregunta **«¿Cómo preparo una tortilla?»**. Verás clasificar → rechazar, sin leer el PDF.
        3. Abre **Observabilidad**, compara tiempos y tokens, y entra en la plataforma MLflow → **Traces**.
        4. Abre una traza y pulsa en sus spans para leer entradas, salidas y duraciones.
        """)
        st.success("Para la entrevista: «He hecho un RAG que recupera fragmentos de un PDF y he instrumentado sus pasos con MLflow para observar decisiones, latencia, tokens y errores». ")
    with st.expander("Relación con LangChain y LangGraph, y documentación"):
        st.write("Usamos el splitter de LangChain. El resto está escrito explícitamente con el SDK de OpenAI y NumPy para que puedas seguir cada paso. LangGraph podría modelar las ramas del clasificador, pero este flujo pequeño no necesita un grafo.")
        st.markdown("""
        - [Structured Outputs de OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs)
        - [Embeddings de OpenAI](https://developers.openai.com/api/docs/guides/embeddings)
        - [Trazas manuales de MLflow](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/)
        - [Persistencia de MLflow](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
        - [Navegación de Streamlit](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation)
        """)
