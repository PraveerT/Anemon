import json
import msvcrt
import os
import ssl
import sys
import threading
import time

import requests
import websocket

from jlab.config import JlabConfig
from jlab.exceptions import ConnectionError


def _get_browser_cookies_and_headers(config: JlabConfig) -> tuple[dict, dict]:
    """Replicate what the browser does: load the JupyterLab page to get
    all session cookies and build proper headers for WebSocket."""
    sess = requests.Session()
    sess.headers.update(config.auth_headers)

    # 1. Hit the main page (like opening the URL in browser)
    sess.get(config.url, verify=False)

    # 2. Hit the /lab page specifically
    sess.get(f"{config.url}/lab", verify=False)

    # 3. Hit the terminals REST API (like clicking Terminal in JupyterLab)
    sess.get(f"{config.api_url}/terminals", verify=False)

    cookies = dict(sess.cookies)

    headers = {
        "Origin": config.url,
        "Referer": f"{config.url}/lab",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    headers.update(config.auth_headers)

    # Add _xsrf as a header too (some Jupyter servers check this)
    if "_xsrf" in cookies:
        headers["X-XSRFToken"] = cookies["_xsrf"]

    return cookies, headers


class TerminalConnection:
    def __init__(self, config: JlabConfig, terminal_name: str,
                 cookies: dict[str, str] | None = None,
                 extra_headers: dict[str, str] | None = None):
        self.config = config
        self.terminal_name = terminal_name
        self.cookies = cookies or {}
        self.extra_headers = extra_headers or {}
        self._ws: websocket.WebSocket | None = None
        self._running = False

    def connect(self) -> None:
        url = (
            f"{self.config.ws_url}/terminals/websocket/{self.terminal_name}"
            f"?token={self.config.token}"
        )
        sslopt = {}
        if url.startswith("wss://"):
            sslopt = {"cert_reqs": ssl.CERT_NONE}

        try:
            # Try minimal headers first (same as kernel WebSocket which works)
            self._ws = websocket.create_connection(
                url,
                header=self.config.auth_headers,
                sslopt=sslopt,
                timeout=30,
            )
        except Exception as e:
            debug_url = url.split("?")[0]
            raise ConnectionError(
                f"Failed to connect to terminal WebSocket.\n"
                f"  URL: {debug_url}\n"
                f"  Error: {type(e).__name__}: {e}"
            )

    def close(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()
            self._ws = None

    def _read_loop(self) -> None:
        while self._running and self._ws:
            try:
                raw = self._ws.recv()
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                msg = json.loads(raw)
                if isinstance(msg, list) and len(msg) >= 2:
                    channel = msg[0]
                    data = msg[1]
                    if channel == "stdout":
                        sys.stdout.write(data)
                        sys.stdout.flush()
            except websocket.WebSocketConnectionClosedException:
                if self._running:
                    sys.stdout.write("\r\nConnection closed by server.\r\n")
                    sys.stdout.flush()
                break
            except json.JSONDecodeError:
                continue
            except Exception:
                if self._running:
                    break

    def _send(self, data: str) -> None:
        if self._ws:
            self._ws.send(json.dumps(["stdin", data]))

    def _send_resize(self, cols: int, rows: int) -> None:
        if self._ws:
            self._ws.send(json.dumps(["set_size", rows, cols]))

    def interactive(self) -> None:
        self._running = True

        try:
            cols, rows = os.get_terminal_size()
            self._send_resize(cols, rows)
        except OSError:
            pass

        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()

        try:
            while self._running:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == '\x00' or ch == '\xe0':
                        ch2 = msvcrt.getwch()
                        arrow_map = {
                            'H': '\x1b[A',  # Up
                            'P': '\x1b[B',  # Down
                            'M': '\x1b[C',  # Right
                            'K': '\x1b[D',  # Left
                            'G': '\x1b[H',  # Home
                            'O': '\x1b[F',  # End
                            'I': '\x1b[5~', # Page Up
                            'Q': '\x1b[6~', # Page Down
                            'S': '\x1b[3~', # Delete
                        }
                        self._send(arrow_map.get(ch2, ''))
                    elif ch == '\x04':
                        break
                    else:
                        self._send(ch)
                else:
                    time.sleep(0.01)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self._running = False
