import json
from dataclasses import dataclass, asdict
from pathlib import Path

from jlab.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".jlab"
CONFIG_FILE = CONFIG_DIR / "config.json"


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
