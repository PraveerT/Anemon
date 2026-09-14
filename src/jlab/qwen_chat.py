"""Durable Qwen chat backed by a localhost-only service on JupyterLab."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click

from jlab.client import JupyterClient
from jlab.config import (
    JlabConfig,
    fetch_running_notebook,
    load_config,
    load_ps_api_key,
    ps_start_notebook,
    save_config,
)
from jlab.kernel import KernelConnection


STATE_DIR = Path.home() / ".jlab" / "qwen"
REMOTE_ROOT = "/notebooks/.anemon-qwen"
DEFAULT_MODEL = "ggml-org/Qwen3.8-27B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.8-27B-Q4_K_M.gguf"
SENTINEL = "__ANEMON_QWEN_RESULT__:"
START_ATTEMPTS = 60
START_INTERVAL = 5
SERVICE_START_TIMEOUT = 1800
REMOTE_SERVER_VERSION = "4"
QWEN_HTTP_TIMEOUT = 4
QWEN_WEBSOCKET_TIMEOUT = 8


class QwenError(RuntimeError):
    """A user-actionable Qwen chat failure."""


@dataclass
class QwenConfig:
    model_id: str = DEFAULT_MODEL
    model_file: str = DEFAULT_MODEL_FILE
    remote_root: str = REMOTE_ROOT
    port: int = 8765
    max_model_len: int = 16384
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024


def _config_file() -> Path:
    return STATE_DIR / "config.json"


def _session_file() -> Path:
    return STATE_DIR / "session.json"


def _history_dir() -> Path:
    return STATE_DIR / "history"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_qwen_config(config: QwenConfig) -> None:
    _atomic_json(_config_file(), asdict(config))


def load_qwen_config() -> QwenConfig:
    path = _config_file()
    if not path.exists():
        return QwenConfig()
    try:
        return QwenConfig(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        raise QwenError(f"Invalid Qwen configuration at {path}: {exc}") from exc


def _validate_chat_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise QwenError(
            "Chat names must be 1-64 characters using letters, numbers, '.', '-', or '_'."
        )
    return name


def history_path(name: str) -> Path:
    return _history_dir() / f"{_validate_chat_name(name)}.json"


def load_history(name: str) -> list[dict[str, str]]:
    path = history_path(name)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QwenError(f"Invalid chat history at {path}: {exc}") from exc
    if not isinstance(data, list) or any(
        not isinstance(item, dict)
        or item.get("role") not in {"system", "user", "assistant"}
        or not isinstance(item.get("content"), str)
        for item in data
    ):
        raise QwenError(f"Invalid chat history format at {path}")
    return data


def save_history(name: str, messages: list[dict[str, str]]) -> None:
    _atomic_json(history_path(name), messages)


def _load_qwen_session() -> dict[str, str] | None:
    path = _session_file()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("kernel_id"):
        return None
    return data


def _save_qwen_session(kernel_id: str, server_url: str) -> None:
    _atomic_json(
        _session_file(),
        {"kernel_id": kernel_id, "server_url": server_url},
    )


def _clear_qwen_session() -> None:
    path = _session_file()
    if path.exists():
        path.unlink()


def _model_directory(config: QwenConfig) -> str:
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.model_id.rsplit("/", 1)[-1])
    digest = hashlib.sha256(config.model_id.encode("utf-8")).hexdigest()[:12]
    return f"{config.remote_root}/models/{basename}-{digest}"


def _model_path(config: QwenConfig) -> str:
    return f"{_model_directory(config)}/{config.model_file}"


def _encoded_payload(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _payload_prelude(value: Any) -> str:
    encoded = _encoded_payload(value)
    return (
        "import base64 as _b64, json as _json\n"
        f"_payload = _json.loads(_b64.b64decode({encoded!r}).decode('utf-8'))\n"
    )


def _extract_sentinel(result: Any) -> Any:
    encoded = None
    for output in result.outputs:
        if output.get("type") != "stream":
            continue
        for line in output.get("text", "").splitlines():
            if line.startswith(SENTINEL):
                encoded = line[len(SENTINEL):]
    if encoded is None:
        detail = result.error_value or "remote bridge returned no result"
        raise QwenError(detail)
    try:
        return json.loads(base64.b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise QwenError("Remote bridge returned an invalid response") from exc


def _result_print_code(expression: str) -> str:
    return (
        f"_result = {expression}\n"
        "_encoded = _b64.b64encode(_json.dumps(_result, ensure_ascii=False).encode('utf-8')).decode('ascii')\n"
        f"print({SENTINEL!r} + _encoded)\n"
    )


class QwenRemote:
    """Owns remote recovery, the dedicated bridge kernel, and inference service."""

    def __init__(self, config: QwenConfig):
        self.config = config
        self.jlab_config: JlabConfig | None = None
        self.client: JupyterClient | None = None

    def connect(self, *, start_if_stopped: bool = True) -> JupyterClient:
        """Use a live saved endpoint or recover it through Paperspace."""
        saved: JlabConfig | None = None
        try:
            saved = load_config()
            client = JupyterClient(saved, request_timeout=QWEN_HTTP_TIMEOUT)
            client.status()
            self.jlab_config = saved
            self.client = client
            return client
        except Exception:
            pass

        api_key = load_ps_api_key()
        if not api_key:
            raise QwenError(
                "The saved JupyterLab endpoint is unavailable and no Paperspace key is saved. "
                "Run 'jlab setup --key <key>'."
            )

        notebook = fetch_running_notebook(api_key)
        if notebook is None:
            if not start_if_stopped:
                raise QwenError("The Paperspace notebook is stopped.")
            ps_start_notebook(api_key)
            for _ in range(START_ATTEMPTS):
                time.sleep(START_INTERVAL)
                notebook = fetch_running_notebook(api_key)
                if notebook is not None:
                    break
            else:
                raise QwenError(
                    "The Paperspace notebook did not become ready in time; check its dashboard."
                )

        refreshed = JlabConfig(
            url=notebook["url"],
            token=notebook["token"],
            default_kernel=saved.default_kernel if saved else "python3",
            default_cwd=saved.default_cwd if saved else "/notebooks",
        )
        client = JupyterClient(refreshed, request_timeout=QWEN_HTTP_TIMEOUT)
        last_error: Exception | None = None
        for _ in range(START_ATTEMPTS):
            try:
                client.status()
                break
            except Exception as exc:
                last_error = exc
                time.sleep(START_INTERVAL)
        else:
            raise QwenError(f"JupyterLab did not become reachable: {last_error}")

        save_config(refreshed)
        self.jlab_config = refreshed
        self.client = client
        return client

    def bridge(self) -> tuple[JupyterClient, str]:
        client = self.client or self.connect()
        assert self.jlab_config is not None
        session = _load_qwen_session()
        if session and session.get("server_url") == self.jlab_config.url:
            try:
                if any(k.id == session["kernel_id"] for k in client.list_kernels()):
                    return client, session["kernel_id"]
            except Exception:
                pass
        _clear_qwen_session()
        kernel = client.start_kernel(self.jlab_config.default_kernel)
        _save_qwen_session(kernel.id, self.jlab_config.url)
        return client, kernel.id

    def _execute(self, code: str, *, timeout: float = 120) -> Any:
        _, kernel_id = self.bridge()
        assert self.jlab_config is not None
        connection = KernelConnection(self.jlab_config, kernel_id)
        try:
            connection.connect(timeout=QWEN_WEBSOCKET_TIMEOUT)
            result = connection.execute(code, timeout=timeout)
        finally:
            connection.close()
        if result.status == "error":
            detail = result.error_value or "remote kernel execution failed"
            raise QwenError(detail)
        return result

    def _execute_streaming(self, code: str, *, timeout: float) -> Any:
        _, kernel_id = self.bridge()
        assert self.jlab_config is not None
        connection = KernelConnection(self.jlab_config, kernel_id)
        try:
            connection.connect(timeout=QWEN_WEBSOCKET_TIMEOUT)
            result = connection.execute_streaming(code, timeout=timeout)
        finally:
            connection.close()
        if result.status == "error":
            detail = result.error_value or "remote kernel execution failed"
            raise QwenError(detail)
        return result

    def invalidate_bridge(self) -> None:
        _clear_qwen_session()
        self.client = None
        self.jlab_config = None

    def provision(self) -> None:
        payload = {
            "root": self.config.remote_root,
            "model_id": self.config.model_id,
            "model_file": self.config.model_file,
            "model_dir": _model_directory(self.config),
            "server_script": REMOTE_SERVER,
        }
        code = _payload_prelude(payload) + PROVISION_CODE
        self._execute_streaming(code, timeout=7200)

    def health(self) -> dict[str, Any]:
        payload = {
            "root": self.config.remote_root,
            "port": self.config.port,
            "model_dir": _model_directory(self.config),
            "model_file": self.config.model_file,
            "model_id": self.config.model_id,
        }
        code = _payload_prelude(payload) + HEALTH_CODE + _result_print_code("_status")
        return _extract_sentinel(self._execute(code, timeout=12))

    def ensure_service(self) -> None:
        status = self.health()
        if (
            status.get("healthy")
            and status.get("service_model") == self.config.model_id
            and status.get("service_version") == REMOTE_SERVER_VERSION
            and status.get("service_max_model_len") == self.config.max_model_len
        ):
            return
        if not status.get("provisioned"):
            raise QwenError("Qwen is not provisioned. Run 'jlab qwen setup' first.")

        payload = {
            "root": self.config.remote_root,
            "port": self.config.port,
            "model_id": self.config.model_id,
            "model_dir": _model_directory(self.config),
            "model_file": self.config.model_file,
            "max_model_len": self.config.max_model_len,
            "timeout": SERVICE_START_TIMEOUT,
            "service_model": status.get("service_model"),
            "restart_service": bool(status.get("healthy")),
        }
        code = _payload_prelude(payload) + START_SERVICE_CODE + _result_print_code("_status")
        result = _extract_sentinel(
            self._execute(code, timeout=SERVICE_START_TIMEOUT + 60)
        )
        if not result.get("healthy"):
            tail = result.get("log_tail", "No service log was produced.")
            raise QwenError(f"Qwen service failed to start. Remote log tail:\n{tail}")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "port": self.config.port,
            "request": {
                "model": self.config.model_id,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
        }
        code = _payload_prelude(payload) + REQUEST_CODE + _result_print_code("_response")
        response = _extract_sentinel(self._execute(code, timeout=1800))
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenError(f"Qwen service returned an unexpected response: {response}") from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenError("Qwen service returned an empty response")
        return content

    def mirror_history(self, name: str, messages: list[dict[str, str]]) -> None:
        payload = {
            "root": self.config.remote_root,
            "name": _validate_chat_name(name),
            "messages": messages,
        }
        self._execute(_payload_prelude(payload) + MIRROR_CODE, timeout=30)


class QwenChat:
    def __init__(self, config: QwenConfig, remote: QwenRemote | None = None):
        self.config = config
        self.remote = remote or QwenRemote(config)

    def send(
        self,
        name: str,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> tuple[str, str | None]:
        """Generate, then commit a complete user/assistant turn atomically."""
        history = load_history(name)
        if system and not any(item["role"] == "system" for item in history):
            history = [{"role": "system", "content": system}, *history]
        pending = [*history, {"role": "user", "content": prompt}]

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                self.remote.connect()
                self.remote.ensure_service()
                answer = self.remote.complete(
                    pending,
                    temperature=self.config.temperature if temperature is None else temperature,
                    top_p=self.config.top_p if top_p is None else top_p,
                    max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    self.remote.invalidate_bridge()
                    continue
                raise QwenError(f"Qwen request failed after recovery: {exc}") from exc
        else:  # pragma: no cover - loop either breaks or raises
            raise QwenError(str(last_error))

        completed = [*pending, {"role": "assistant", "content": answer}]
        save_history(name, completed)
        mirror_warning = None
        try:
            self.remote.mirror_history(name, completed)
        except Exception as exc:
            mirror_warning = f"Remote history mirror failed; local history is safe: {exc}"
        return answer, mirror_warning


REMOTE_SERVER = r'''#!/usr/bin/env python3
import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_cpp import Llama

MODEL_ID = os.environ["ANEMON_QWEN_MODEL_ID"]
MODEL_PATH = os.environ["ANEMON_QWEN_MODEL_PATH"]
PORT = int(os.environ["ANEMON_QWEN_PORT"])
MAX_MODEL_LEN = int(os.environ.get("ANEMON_QWEN_MAX_MODEL_LEN", "16384"))
SERVER_VERSION = "4"

print(f"Loading {MODEL_ID} from {MODEL_PATH} with llama.cpp...", flush=True)
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_MODEL_LEN,
    n_gpu_layers=-1,
    verbose=True,
)
generation_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "model": MODEL_ID,
                "version": SERVER_VERSION,
                "max_model_len": MAX_MODEL_LEN,
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            messages = request["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            max_tokens = max(1, int(request.get("max_tokens", 1024)))
            temperature = float(request.get("temperature", 0.6))
            with generation_lock:
                completion = model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=float(request.get("top_p", 0.95)),
                    chat_template_kwargs={"enable_thinking": False},
                )
            answer = completion["choices"][0]["message"]["content"] or ""
            # Defense in depth for templates that still emit a reasoning block.
            answer = re.sub(r"(?s)^.*?</think>\s*", "", answer).strip()
            self._json(200, {
                "id": "anemon-qwen",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
            })
        except Exception as exc:
            traceback.print_exc()
            self._json(400, {"error": {"message": str(exc), "type": type(exc).__name__}})

    def log_message(self, *args):
        return


print(f"Serving on 127.0.0.1:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


PROVISION_CODE = r'''
import os as _os, pathlib as _pathlib, subprocess as _subprocess, sys as _sys
_root = _pathlib.Path(_payload["root"])
_venv = _root / "venv-cu124"
_python = _venv / "bin" / "python"
_model_dir = _pathlib.Path(_payload["model_dir"])
_root.mkdir(parents=True, exist_ok=True)
(_root / "models").mkdir(exist_ok=True)
(_root / "hf-cache").mkdir(exist_ok=True)
print(f"Persistent Qwen root: {_root}", flush=True)
if not _python.exists():
    print("Creating isolated Python environment...", flush=True)
    _subprocess.run([_sys.executable, "-m", "venv", str(_venv)], check=True)
_deps_marker = _root / ".deps-torch26-cu124-transformers515-v2"
if not _deps_marker.exists():
    print("Installing PyTorch 2.6.0 (CUDA 12.4) in the isolated environment...", flush=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install", "--upgrade", "pip"
    ], check=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "torch==2.6.0", "torchvision==0.21.0",
    ], check=True)
    print("Installing Transformers, Accelerate, bitsandbytes, and Hugging Face Hub...", flush=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install",
        "transformers>=5.15,<5.16", "accelerate", "bitsandbytes", "huggingface_hub",
        "pydantic>=2", "loguru",
    ], check=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install", "--no-deps", "compressed-tensors==0.17.0"
    ], check=True)
    _deps_marker.write_text("torch=2.6.0+cu124\ntransformers>=5.15,<5.16\n", encoding="utf-8")
else:
    print("Dependencies already provisioned; skipping install.", flush=True)
_compression_version = _subprocess.run(
    [str(_python), "-c", "import importlib.metadata as m; print(m.version('compressed-tensors'))"],
    capture_output=True, text=True,
)
if _compression_version.returncode != 0 or _compression_version.stdout.strip() != "0.17.0":
    print("Updating compressed-tensors compatibility layer...", flush=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install", "--no-deps", "--upgrade",
        "compressed-tensors==0.17.0",
    ], check=True)
_llama_version = _subprocess.run(
    [str(_python), "-c", "import importlib.metadata as m; print(m.version('llama-cpp-python'))"],
    capture_output=True, text=True,
)
if _llama_version.returncode != 0 or _llama_version.stdout.strip() != "0.3.35":
    print("Installing CUDA 12.4 llama.cpp runtime...", flush=True)
    _subprocess.run([
        str(_python), "-m", "pip", "install", "--only-binary=:all:",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        "llama-cpp-python==0.3.35",
    ], check=True)
_server = _root / "server.py"
if not _server.exists() or _server.read_text(encoding="utf-8") != _payload["server_script"]:
    _server.write_text(_payload["server_script"], encoding="utf-8")
_complete = _model_dir / ".anemon-complete"
_completion_value = _payload["model_id"] + "\n" + _payload["model_file"]
if _complete.exists() and _complete.read_text(encoding="utf-8").strip() == _completion_value:
    print(f"Model snapshot already complete at {_model_dir}; skipping download.", flush=True)
else:
    print(f"Downloading {_payload['model_id']}:{_payload['model_file']} to {_model_dir}...", flush=True)
    _model_dir.mkdir(parents=True, exist_ok=True)
    _download = (
        "from huggingface_hub import hf_hub_download; import sys; "
        "hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], "
        "local_dir=sys.argv[3], cache_dir=sys.argv[4])"
    )
    _subprocess.run([
        str(_python), "-c", _download, _payload["model_id"], _payload["model_file"],
        str(_model_dir), str(_root / "hf-cache")
    ], check=True)
    _complete.write_text(_completion_value + "\n", encoding="utf-8")
print("Qwen provisioning complete.", flush=True)
'''


HEALTH_CODE = r'''
import os as _os, pathlib as _pathlib, urllib.request as _urlrequest
_root = _pathlib.Path(_payload["root"])
_model_dir = _pathlib.Path(_payload["model_dir"])
_complete = _model_dir / ".anemon-complete"
_completion_value = _payload["model_id"] + "\n" + _payload["model_file"]
_provisioned = (
    (_root / "venv-cu124" / "bin" / "python").exists()
    and (_root / "server.py").exists()
    and _complete.exists()
    and _complete.read_text(encoding="utf-8").strip() == _completion_value
)
_healthy = False
_service_model = None
_service_version = None
_service_max_model_len = None
try:
    with _urlrequest.urlopen(f"http://127.0.0.1:{_payload['port']}/health", timeout=2) as _response:
        _health = _json.loads(_response.read().decode("utf-8"))
        _healthy = _response.status == 200 and _health.get("status") == "ok"
        _service_model = _health.get("model")
        _service_version = _health.get("version")
        _service_max_model_len = _health.get("max_model_len")
except Exception:
    pass
_status = {
    "healthy": _healthy,
    "provisioned": _provisioned,
    "model": _payload["model_id"],
    "service_model": _service_model,
    "service_version": _service_version,
    "service_max_model_len": _service_max_model_len,
    "root": str(_root),
}
'''


START_SERVICE_CODE = r'''
import os as _os, pathlib as _pathlib, signal as _signal, subprocess as _subprocess, time as _time, urllib.request as _urlrequest
_root = _pathlib.Path(_payload["root"])
_pid_path = _root / "server.pid"
_log_path = _root / "server.log"

def _is_healthy():
    try:
        with _urlrequest.urlopen(f"http://127.0.0.1:{_payload['port']}/health", timeout=2) as _response:
            return _response.status == 200
    except Exception:
        return False

def _pid_alive(pid):
    try:
        _state = _pathlib.Path(f"/proc/{pid}/stat").read_text().split()[2]
        if _state == "Z":
            return False
        _os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False

_pid = None
if _pid_path.exists():
    try:
        _pid = int(_pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        _pid = None
if _payload.get("restart_service"):
    if _pid is None or not _pid_alive(_pid):
        raise RuntimeError(
            f"Port {_payload['port']} has a service that is not owned by Anemon"
        )
    _os.kill(_pid, _signal.SIGTERM)
    for _ in range(30):
        if not _pid_alive(_pid):
            break
        _time.sleep(1)
    if _pid_alive(_pid):
        raise RuntimeError(
            f"The existing Qwen service (PID {_pid}) did not stop; refusing to launch a duplicate"
        )
    _pid = None
if _pid is None or not _pid_alive(_pid):
    _env = _os.environ.copy()
    _env.update({
        "ANEMON_QWEN_MODEL_ID": _payload["model_id"],
        "ANEMON_QWEN_MODEL_PATH": str(_pathlib.Path(_payload["model_dir"]) / _payload["model_file"]),
        "ANEMON_QWEN_PORT": str(_payload["port"]),
        "ANEMON_QWEN_MAX_MODEL_LEN": str(_payload["max_model_len"]),
        "HF_HOME": str(_root / "hf-cache"),
        "PYTHONUNBUFFERED": "1",
    })
    _log = _log_path.open("ab", buffering=0)
    _process = _subprocess.Popen(
        [str(_root / "venv-cu124" / "bin" / "python"), str(_root / "server.py")],
        cwd=str(_root), stdout=_log, stderr=_subprocess.STDOUT,
        env=_env, start_new_session=True,
    )
    _pid = _process.pid
    _temporary = _pid_path.with_suffix(".tmp")
    _temporary.write_text(str(_pid), encoding="utf-8")
    _os.replace(_temporary, _pid_path)
_deadline = _time.time() + int(_payload["timeout"])
while _time.time() < _deadline and _pid_alive(_pid):
    if _is_healthy():
        break
    _time.sleep(5)
try:
    _tail = _log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
except OSError:
    _tail = ""
_status = {"healthy": _is_healthy(), "pid": _pid, "log_tail": _tail}
'''


REQUEST_CODE = r'''
import urllib.error as _urlerror, urllib.request as _urlrequest
_body = _json.dumps(_payload["request"], ensure_ascii=False).encode("utf-8")
_request = _urlrequest.Request(
    f"http://127.0.0.1:{_payload['port']}/v1/chat/completions",
    data=_body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with _urlrequest.urlopen(_request, timeout=1700) as _remote_response:
        _response = _json.loads(_remote_response.read().decode("utf-8"))
except _urlerror.HTTPError as _exc:
    _detail = _exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"Qwen service HTTP {_exc.code}: {_detail}") from _exc
'''


MIRROR_CODE = r'''
import os as _os, pathlib as _pathlib
_directory = _pathlib.Path(_payload["root"]) / "history"
_directory.mkdir(parents=True, exist_ok=True)
_target = _directory / (_payload["name"] + ".json")
_temporary = _target.with_suffix(".tmp")
_temporary.write_text(
    _json.dumps(_payload["messages"], indent=2, ensure_ascii=False), encoding="utf-8"
)
_os.replace(_temporary, _target)
'''


def _raise_click_error(exc: Exception) -> None:
    if isinstance(exc, click.ClickException):
        raise exc
    raise click.ClickException(str(exc)) from exc


@click.group()
def qwen() -> None:
    """Run durable Qwen chat on the remote Paperspace GPU."""


@qwen.command("setup")
@click.option("--model", "model_id", default=None, help=f"Model ID (default: {DEFAULT_MODEL})")
@click.option("--model-file", default=None, help=f"GGUF filename (default: {DEFAULT_MODEL_FILE})")
@click.option("--port", type=click.IntRange(1024, 65535), default=None, help="Remote localhost port")
@click.option("--max-model-len", type=click.IntRange(1024), default=None, help="Maximum context length")
def setup_command(
    model_id: str | None,
    model_file: str | None,
    port: int | None,
    max_model_len: int | None,
) -> None:
    """Create the remote venv and download the durable model snapshot."""
    try:
        config = load_qwen_config()
        if model_id is not None:
            config.model_id = model_id
        if model_file is not None:
            config.model_file = model_file
        if port is not None:
            config.port = port
        if max_model_len is not None:
            config.max_model_len = max_model_len
        save_qwen_config(config)
        click.echo(f"Model: {config.model_id}")
        click.echo(f"Remote root: {config.remote_root}")
        click.echo("Provisioning may take a long time on the first run...")
        remote = QwenRemote(config)
        remote.connect()
        remote.provision()
        click.echo("Qwen setup complete.")
    except Exception as exc:
        _raise_click_error(exc)


@qwen.command("chat")
@click.argument("prompt", required=False)
@click.option("--chat", "chat_name", default="default", show_default=True, help="Durable conversation name")
@click.option("--temperature", type=click.FloatRange(0, 2), default=None, help="Sampling temperature")
@click.option("--top-p", type=click.FloatRange(0, 1, min_open=True), default=None, help="Nucleus sampling threshold")
@click.option("--max-tokens", type=click.IntRange(1), default=None, help="Maximum new tokens")
@click.option("--system", default=None, help="System message for a new conversation")
def chat_command(
    prompt: str | None,
    chat_name: str,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    system: str | None,
) -> None:
    """Send PROMPT once, or omit it for an interactive chat loop."""
    try:
        config = load_qwen_config()
        chat = QwenChat(config)

        def send(text: str) -> None:
            answer, warning = chat.send(
                chat_name,
                text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system=system,
            )
            click.echo(answer)
            if warning:
                click.echo(f"Warning: {warning}", err=True)

        if prompt is not None:
            send(prompt)
            return

        click.echo(f"Qwen chat '{chat_name}'. Type /exit or press Ctrl+D to leave.")
        while True:
            try:
                text = click.prompt("You", prompt_suffix="> ")
            except (EOFError, click.Abort):
                click.echo()
                return
            if text.strip().lower() in {"/exit", "/quit"}:
                return
            if text.strip():
                send(text)
    except Exception as exc:
        _raise_click_error(exc)


@qwen.command("status")
def status_command() -> None:
    """Show local configuration and remote provisioning/service state."""
    config = load_qwen_config()
    click.echo(f"Model: {config.model_id}")
    click.echo(f"Remote root: {config.remote_root}")
    click.echo(f"Local state: {STATE_DIR}")
    remote = QwenRemote(config)
    try:
        remote.connect(start_if_stopped=False)
        # An interrupted command can leave its dedicated bridge busy. Status is
        # diagnostic, so use a fresh bounded bridge instead of waiting on it.
        _clear_qwen_session()
        status = remote.health()
    except Exception as exc:
        click.echo(f"Remote: unavailable ({exc})")
        return
    click.echo("Remote: connected")
    click.echo(f"Provisioned: {'yes' if status.get('provisioned') else 'no'}")
    click.echo(f"Service: {'ready' if status.get('healthy') else 'stopped'}")
    if status.get("service_model"):
        click.echo(f"Loaded model: {status['service_model']}")


@qwen.command("history")
@click.option("--chat", "chat_name", default="default", show_default=True, help="Conversation name")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON")
def history_command(chat_name: str, as_json: bool) -> None:
    """Print a durable local conversation transcript."""
    try:
        messages = load_history(chat_name)
    except Exception as exc:
        _raise_click_error(exc)
        return
    if as_json:
        click.echo(json.dumps(messages, indent=2, ensure_ascii=False))
        return
    if not messages:
        click.echo(f"No history for chat '{chat_name}'.")
        return
    for message in messages:
        click.echo(f"{message['role'].capitalize()}: {message['content']}")
