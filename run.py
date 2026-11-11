"""Arranca Streamlit y la UI de MLflow con una sola orden, también en Windows."""

from pathlib import Path
from time import monotonic, sleep
import argparse
import os
import signal
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def port_is_free(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", port)) != 0


def stop_services(children):
    """Cierra cada servicio con sus procesos secundarios en Windows y POSIX."""
    for child in children:
        if os.name == "nt":
            if child.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
        else:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                child.kill()
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()
    for port in (5000, 8501):
        if not port_is_free(port):
            print(f"El puerto {port} ya está ocupado. Cierra la instancia anterior antes de arrancar.")
            return 1

    from quixote.config import load_settings
    from quixote.telemetry import configure_mlflow
    settings = load_settings()
    configure_mlflow(settings)
    commands = [
        [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1",
         "--port", "5000", "--workers", "1", "--backend-store-uri", settings.tracking_uri,
         "--default-artifact-root", (settings.runtime / "artifacts").as_uri()],
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true"],
    ]
    children, logs = [], []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for name, command in zip(("mlflow", "streamlit"), commands, strict=True):
            log = (settings.runtime / f"{name}.log").open("a", encoding="utf-8")
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                             creationflags=flags, start_new_session=os.name != "nt"))
        deadline = monotonic() + 90
        urls = ["http://127.0.0.1:5000/health", "http://127.0.0.1:8501/_stcore/health"]
        pending = set(urls)
        while pending:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Un servicio se cerró. Revisa los archivos .runtime/mlflow.log y .runtime/streamlit.log.")
            for url in list(pending):
                try:
                    with urllib.request.urlopen(url, timeout=1) as response:
                        if response.status == 200:
                            pending.remove(url)
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
            if monotonic() > deadline:
                raise RuntimeError("Los servicios no arrancaron a tiempo. Revisa los logs de .runtime/.")
            sleep(.3)
        print("Chat:   http://127.0.0.1:8501", flush=True)
        print("MLflow: http://127.0.0.1:5000", flush=True)
        print("Pulsa Ctrl+C para detener ambos servicios.", flush=True)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:8501")
        while all(child.poll() is None for child in children):
            sleep(.5)
        raise RuntimeError("Uno de los servicios se ha detenido. Revisa los logs en .runtime/.")
    except KeyboardInterrupt:
        print("\nCerrando la biblioteca…")
        return 0
    except RuntimeError as exc:
        print(str(exc))
        return 1
    finally:
        try:
            stop_services(children)
        finally:
            for log in logs:
                log.close()


if __name__ == "__main__":
    raise SystemExit(main())
