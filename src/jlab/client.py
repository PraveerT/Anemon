import base64
import json
from pathlib import Path

import requests

from jlab.config import JlabConfig
from jlab.exceptions import AuthenticationError, ConnectionError, NotFoundError
from jlab.models import ContentItem, ContentType, KernelInfo, ServerStatus, SessionInfo


class JupyterClient:
    def __init__(self, config: JlabConfig):
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(config.auth_headers)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.config.api_url}/{path.lstrip('/')}"
        try:
            resp = self._session.request(method, url, **kwargs)
        except requests.ConnectionError:
            raise ConnectionError(f"Cannot reach server at {self.config.url}")
        if resp.status_code == 401:
            raise AuthenticationError("Invalid token")
        if resp.status_code == 403:
            raise AuthenticationError("Access forbidden - check your token")
        if resp.status_code == 404:
            raise NotFoundError(f"Not found: {path}")
        resp.raise_for_status()
        return resp

    def _parse_content_item(self, data: dict) -> ContentItem:
        return ContentItem(
            name=data["name"],
            path=data["path"],
            type=ContentType(data["type"]),
            size=data.get("size"),
            last_modified=data.get("last_modified", ""),
            writable=data.get("writable", True),
            content=data.get("content"),
            format=data.get("format"),
            mimetype=data.get("mimetype"),
        )

    def _parse_kernel_info(self, data: dict) -> KernelInfo:
        return KernelInfo(
            id=data["id"],
            name=data["name"],
            last_activity=data.get("last_activity", ""),
            execution_state=data.get("execution_state", "unknown"),
            connections=data.get("connections", 0),
        )

    # --- Server ---

    def status(self) -> ServerStatus:
        resp = self._request("GET", "status")
        data = resp.json()
        return ServerStatus(
            started=data.get("started", ""),
            last_activity=data.get("last_activity", ""),
            connections=data.get("connections", 0),
            kernels=data.get("kernels", 0),
        )

    # --- Contents ---

    def list_contents(self, path: str = "") -> list[ContentItem]:
        resp = self._request("GET", f"contents/{path}", params={"content": 1})
        data = resp.json()
        if data["type"] == "directory":
            return [self._parse_content_item(item) for item in data.get("content", [])]
        return [self._parse_content_item(data)]

    def get_contents(self, path: str) -> ContentItem:
        resp = self._request("GET", f"contents/{path}", params={"content": 1})
        return self._parse_content_item(resp.json())

    def upload_file(self, local_path: Path, remote_path: str) -> ContentItem:
        raw = local_path.read_bytes()

        if local_path.suffix == ".ipynb":
            body = {
                "content": json.loads(raw),
                "format": "json",
                "type": "notebook",
            }
        else:
            try:
                text = raw.decode("utf-8")
                body = {"content": text, "format": "text", "type": "file"}
            except UnicodeDecodeError:
                body = {
                    "content": base64.b64encode(raw).decode("ascii"),
                    "format": "base64",
                    "type": "file",
                }

        resp = self._request("PUT", f"contents/{remote_path}", json=body)
        return self._parse_content_item(resp.json())

    def download_file(self, remote_path: str) -> tuple[str | bytes, str]:
        resp = self._request("GET", f"contents/{remote_path}", params={"content": 1})
        data = resp.json()
        fmt = data.get("format", "text")
        if fmt == "base64":
            return base64.b64decode(data["content"]), fmt
        elif fmt == "json":
            return json.dumps(data["content"], indent=2), fmt
        else:
            return data["content"], fmt

    def delete(self, path: str) -> None:
        self._request("DELETE", f"contents/{path}")

    # --- Kernels ---

    def list_kernels(self) -> list[KernelInfo]:
        resp = self._request("GET", "kernels")
        return [self._parse_kernel_info(k) for k in resp.json()]

    def start_kernel(self, name: str) -> KernelInfo:
        resp = self._request("POST", "kernels", json={"name": name})
        return self._parse_kernel_info(resp.json())

    def delete_kernel(self, kernel_id: str) -> None:
        self._request("DELETE", f"kernels/{kernel_id}")

    def restart_kernel(self, kernel_id: str) -> None:
        self._request("POST", f"kernels/{kernel_id}/restart")

    # --- Sessions ---

    def list_sessions(self) -> list[SessionInfo]:
        resp = self._request("GET", "sessions")
        sessions = []
        for s in resp.json():
            kernel = self._parse_kernel_info(s["kernel"])
            sessions.append(SessionInfo(
                id=s["id"],
                path=s["path"],
                name=s["name"],
                type=s["type"],
                kernel=kernel,
            ))
        return sessions

    # --- Terminals ---

    def create_terminal(self) -> dict:
        resp = self._request("POST", "terminals")
        return resp.json()

    def delete_terminal(self, name: str) -> None:
        self._request("DELETE", f"terminals/{name}")

    # --- Kernel Specs ---

    def get_kernelspecs(self) -> dict:
        resp = self._request("GET", "kernelspecs")
        return resp.json()
