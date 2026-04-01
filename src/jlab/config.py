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
    Returns {"url": ..., "token": ..., "id": ..., "name": ...} or None."""
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
                "id": nb["id"],
                "name": nb.get("name", ""),
                "projectId": nb.get("projectId", ""),
            }
    return None


# Default container and project for starting notebooks
PS_LEGACY_API = "https://api.paperspace.io/notebooks/v2"
PS_DEFAULT_CONTAINER = "nvcr.io/nvidia/pytorch:23.10-py3"


PS_NOTEBOOK_ID = "nfjbqnsvpx"
PS_PROJECT_ID = "pitgq1c6bcy"


def ps_start_notebook(api_key: str) -> dict:
    """Start notebook nfjbqnsvpx with Free-A6000. Fails if unavailable."""
    import requests
    resp = requests.post(
        f"{PS_LEGACY_API}/createNotebook",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={
            "machineType": "Free-A6000",
            "projectId": PS_PROJECT_ID,
            "name": "PMamba",
            "container": PS_DEFAULT_CONTAINER,
            "shutdownTimeout": 6,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise ConfigError(f"Failed to start notebook: {resp.text[:200]}")
    data = resp.json()
    return {
        "id": data["handle"],
        "url": f"https://{data['fqdn']}",
        "token": data["token"],
        "state": data.get("state", "Pending"),
    }


def ps_stop_notebook(api_key: str) -> None:
    """Stop the running notebook."""
    import requests
    nb = fetch_running_notebook(api_key)
    if not nb:
        return
    requests.post(
        f"{PS_LEGACY_API}/stopNotebook",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"notebookId": nb["id"]},
        timeout=30,
    ).raise_for_status()
