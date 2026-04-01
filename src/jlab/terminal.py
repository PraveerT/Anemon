import json
import ssl
import sys
import threading

import websocket

from jlab.config import JlabConfig
from jlab.exceptions import ConnectionError


class TerminalConnection:
    def __init__(self, config: JlabConfig, terminal_name: str):
        self.config = config
        self.terminal_name = terminal_name
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

        headers = dict(self.config.auth_headers)
        headers["Origin"] = self.config.url

        try:
            self._ws = websocket.create_connection(
                url,
                header=headers,
                sslopt=sslopt,
                timeout=30,
            )
        except Exception as e:
            # Show the URL (without token) for debugging
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
        """Background thread: read output from remote terminal and print it."""
        while self._running and self._ws:
            try:
                raw = self._ws.recv()
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
            except Exception:
                if self._running:
                    break

    def _send(self, data: str) -> None:
        if self._ws:
            self._ws.send(json.dumps(["stdin", data]))

    def interactive(self) -> None:
        """Run an interactive shell session."""
        self._running = True

        # Start background thread to read output
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()

        try:
            while self._running:
                try:
                    line = input()
                    self._send(line + "\n")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    # Send Ctrl+C to remote
                    self._send("\x03")
        finally:
            self._running = False
