import json
from dataclasses import dataclass, asdict
from pathlib import Path

from jlab.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".jlab"
CONFIG_FILE = CONFIG_DIR / "config.json"
PS_API_KEY_FILE = CONFIG_DIR / "paperspace_key"


@dataclass
class JlabConfig:
    url: str
    token: str
    default_kernel: str = "python3"

    @property
    def api_url(self) -> str:
        return self.url.rstrip("/") + "/api"

    @property
    def ws_url(self) -> str:
        return self.api_url.replace("https://", "wss://").replace("http://", "ws://")

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}


def save_config(config: JlabConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(asdict(config), indent=2))


def load_config() -> JlabConfig:
    if not CONFIG_FILE.exists():
        raise ConfigError("Not connected. Run 'jlab connect <url> --token <token>' first.")
    data = json.loads(CONFIG_FILE.read_text())
    return JlabConfig(**data)


# --- Persistent session ---

SESSION_FILE = CONFIG_DIR / "session.json"


def save_session(kernel_id: str, cwd: str = "/notebooks") -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps({"kernel_id": kernel_id, "cwd": cwd}))


def load_session() -> dict | None:
    if not SESSION_FILE.exists():
        return None
    try:
        return json.loads(SESSION_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def clear_session() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


# --- Paperspace API ---

def save_ps_api_key(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PS_API_KEY_FILE.write_text(key)


def load_ps_api_key() -> str | None:
    if not PS_API_KEY_FILE.exists():
        return None
    return PS_API_KEY_FILE.read_text().strip()


def fetch_running_notebook(api_key: str) -> dict | None:
    """Query Paperspace API for a running Gradient notebook.
    Returns {"url": ..., "token": ...} or None."""
    import requests
    resp = requests.get(
        "https://api.paperspace.com/v1/notebooks",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    for nb in resp.json().get("items", []):
        if nb.get("state") == "Running" and nb.get("fqdn") and nb.get("token"):
            return {
                "url": f"https://{nb['fqdn']}",
                "token": nb["token"],
                "name": nb.get("name", ""),
            }
    return None
