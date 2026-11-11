"""El lanzador debe liberar los puertos que abren los procesos secundarios."""

from time import monotonic, sleep
import os
import socket
import subprocess
import sys

from run import port_is_free, stop_services


def test_port_probe_detects_a_running_service():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not port_is_free(port)
    assert port_is_free(port)


def test_shutdown_closes_descendant_listener(tmp_path):
    port_file = tmp_path / "port.txt"
    listener_code = """
from pathlib import Path
import socket, sys, time
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    Path(sys.argv[1]).write_text(str(listener.getsockname()[1]), encoding='utf-8')
    time.sleep(60)
"""
    parent_code = """
import subprocess, sys
subprocess.run([sys.executable, '-c', sys.argv[1], sys.argv[2]], check=True)
"""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code, listener_code, str(port_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )
    try:
        deadline = monotonic() + 15
        while not port_file.exists() and process.poll() is None and monotonic() < deadline:
            sleep(.05)
        assert port_file.exists(), "El proceso secundario no pudo abrir el puerto de prueba."
        port = int(port_file.read_text(encoding="utf-8"))
        assert not port_is_free(port)
        stop_services([process])
        deadline = monotonic() + 5
        while not port_is_free(port) and monotonic() < deadline:
            sleep(.05)
        assert process.poll() is not None
        assert port_is_free(port), "El proceso secundario dejó el puerto ocupado."
    finally:
        stop_services([process])
