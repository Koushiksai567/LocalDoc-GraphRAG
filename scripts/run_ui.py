from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
HOST = os.getenv("GRAPHRAG_UI_BACKEND_HOST", "127.0.0.1")
PORT = int(os.getenv("GRAPHRAG_UI_BACKEND_PORT", "8000"))
UI_PORT = int(os.getenv("GRAPHRAG_UI_PORT", "8501"))
API_URL = f"http://{HOST}:{PORT}"


def _load_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in _load_dotenv().items():
        env.setdefault(key, value)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    env["GRAPHRAG_API_URL"] = API_URL
    return env


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        return
    example = PROJECT_ROOT / ".env.example"
    if not example.exists():
        raise RuntimeError("Neither .env nor .env.example exists.")
    env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example")


def _ensure_neo4j() -> None:
    if _port_open("127.0.0.1", 7687):
        print("Neo4j is already reachable on port 7687.")
        return
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not installed or is not available in PATH.")
    print("Starting Neo4j with Docker...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "neo4j"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Could not start Neo4j with Docker:\n{detail}")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", 7687):
            print("Neo4j is ready.")
            return
        time.sleep(2)
    raise RuntimeError("Neo4j did not become reachable on port 7687 within two minutes.")


def _ollama_models() -> set[str]:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return set()
    return {
        str(item.get("name", ""))
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }


def _ensure_ollama(env: dict[str, str]) -> None:
    models = _ollama_models()
    if not models and sys.platform == "darwin":
        print("Starting the Ollama application...")
        subprocess.run(["open", "-a", "Ollama"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            models = _ollama_models()
            if models or _http_ok("http://127.0.0.1:11434/api/tags"):
                break
            time.sleep(1)
    if not _http_ok("http://127.0.0.1:11434/api/tags"):
        raise RuntimeError(
            "Ollama is not running on http://127.0.0.1:11434. Open the Ollama application and run this command again."
        )

    required_model = env.get("GENERATION_MODEL", "qwen3:1.7b")
    models = _ollama_models()
    if required_model in models:
        print(f"Ollama model is ready: {required_model}")
        return
    if shutil.which("ollama") is None:
        raise RuntimeError(
            f"The required model {required_model} is missing. Install the Ollama CLI and run: ollama pull {required_model}"
        )
    print(f"Downloading missing Ollama model: {required_model}")
    result = subprocess.run(["ollama", "pull", required_model], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not download Ollama model {required_model}.")


def _wait_for_backend(process: subprocess.Popen[bytes], timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "The GraphRAG backend stopped during startup. Review the error printed above in this terminal."
            )
        if _http_ok(f"{API_URL}/health", timeout=3):
            print(f"GraphRAG backend is ready at {API_URL}")
            return
        time.sleep(1)
    raise RuntimeError("The GraphRAG backend did not become ready within three minutes.")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    _ensure_env_file()
    env = _project_env()
    backend_process: subprocess.Popen[bytes] | None = None

    try:
        _ensure_neo4j()
        _ensure_ollama(env)

        if _http_ok(f"{API_URL}/health"):
            print(f"Using the existing backend at {API_URL}")
        else:
            if _port_open(HOST, PORT):
                raise RuntimeError(
                    f"Port {PORT} is already in use by another process. Stop that process and run the launcher again."
                )
            print("Starting the GraphRAG backend...")
            backend_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "enterprise_graphrag.main:app",
                    "--app-dir",
                    str(SRC_DIR),
                    "--host",
                    HOST,
                    "--port",
                    str(PORT),
                    "--log-level",
                    env.get("LOG_LEVEL", "info").lower(),
                ],
                cwd=PROJECT_ROOT,
                env=env,
            )
            _wait_for_backend(backend_process)

        print(f"Starting Streamlit at http://localhost:{UI_PORT}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(PROJECT_ROOT / "streamlit_app.py"),
                "--server.port",
                str(UI_PORT),
                "--server.headless=false",
                "--browser.gatherUsageStats=false",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print("\nStopping Local GraphRAG...")
    except Exception as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if backend_process is not None and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()


if __name__ == "__main__":
    main()
