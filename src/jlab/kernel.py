import json
import ssl
import sys
import time
import uuid
from datetime import datetime, timezone

import websocket

from jlab.config import JlabConfig
from jlab.exceptions import KernelError
from jlab.models import ExecutionResult


class KernelConnection:
    def __init__(self, config: JlabConfig, kernel_id: str):
        self.config = config
        self.kernel_id = kernel_id
        self.session_id = str(uuid.uuid4())
        self._ws: websocket.WebSocket | None = None

    def connect(self) -> None:
        url = (
            f"{self.config.ws_url}/kernels/{self.kernel_id}/channels"
            f"?token={self.config.token}"
        )
        sslopt = {}
        if url.startswith("wss://"):
            sslopt = {"cert_reqs": ssl.CERT_NONE}
        self._ws = websocket.create_connection(
            url,
            header=self.config.auth_headers,
            sslopt=sslopt,
            timeout=30,
        )

    def close(self) -> None:
        if self._ws:
            self._ws.close()
            self._ws = None

    def _make_header(self, msg_type: str) -> dict:
        return {
            "msg_id": str(uuid.uuid4()),
            "session": self.session_id,
            "username": "jlab-cli",
            "date": datetime.now(timezone.utc).isoformat(),
            "msg_type": msg_type,
            "version": "5.3",
        }

    def _make_execute_request(self, code: str) -> dict:
        header = self._make_header("execute_request")
        return {
            "header": header,
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }

    def execute(self, code: str, timeout: float = 120.0) -> ExecutionResult:
        if not self._ws:
            raise KernelError("Not connected to kernel")

        msg = self._make_execute_request(code)
        msg_id = msg["header"]["msg_id"]
        self._ws.send(json.dumps(msg))

        result = ExecutionResult(status="ok")
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._ws.settimeout(remaining)
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            except (websocket.WebSocketConnectionClosedException, OSError, ConnectionError) as e:
                raise KernelError(f"Connection lost: {e}")

            # Skip binary frames
            if isinstance(raw, bytes):
                continue

            try:
                reply = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            parent_msg_id = reply.get("parent_header", {}).get("msg_id")
            if parent_msg_id != msg_id:
                continue

            msg_type = reply["header"]["msg_type"]
            content = reply.get("content", {})

            match msg_type:
                case "stream":
                    result.outputs.append({
                        "type": "stream",
                        "name": content["name"],
                        "text": content["text"],
                    })
                case "execute_result":
                    result.execution_count = content.get("execution_count", 0)
                    result.outputs.append({
                        "type": "execute_result",
                        "data": content["data"],
                    })
                case "display_data":
                    result.outputs.append({
                        "type": "display_data",
                        "data": content["data"],
                    })
                case "error":
                    result.status = "error"
                    result.error_name = content["ename"]
                    result.error_value = content["evalue"]
                    result.traceback = content.get("traceback", [])
                case "execute_reply":
                    result.status = content["status"]
                    if content["status"] == "error":
                        result.error_name = content.get("ename", "")
                        result.error_value = content.get("evalue", "")
                        result.traceback = content.get("traceback", [])
                    return result
                case "status":
                    pass

        raise KernelError("Execution timed out")

    def execute_streaming(self, code: str, timeout: float = 120.0) -> ExecutionResult:
        """Execute code and print stream output in real-time as it arrives."""
        if not self._ws:
            raise KernelError("Not connected to kernel")

        msg = self._make_execute_request(code)
        msg_id = msg["header"]["msg_id"]
        self._ws.send(json.dumps(msg))

        result = ExecutionResult(status="ok")
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._ws.settimeout(remaining)
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            except (websocket.WebSocketConnectionClosedException, OSError, ConnectionError) as e:
                raise KernelError(f"Connection lost: {e}")

            if isinstance(raw, bytes):
                continue

            try:
                reply = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            parent_msg_id = reply.get("parent_header", {}).get("msg_id")
            if parent_msg_id != msg_id:
                continue

            msg_type = reply["header"]["msg_type"]
            content = reply.get("content", {})

            match msg_type:
                case "stream":
                    text = content["text"]
                    try:
                        if content["name"] == "stderr":
                            sys.stderr.write(text)
                            sys.stderr.flush()
                        else:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                    except UnicodeEncodeError:
                        # Windows cp1252 can't encode some Unicode chars
                        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace")
                        sys.stdout.write(encoded)
                        sys.stdout.flush()
                    result.outputs.append({
                        "type": "stream",
                        "name": content["name"],
                        "text": text,
                    })
                case "execute_result":
                    result.execution_count = content.get("execution_count", 0)
                    result.outputs.append({
                        "type": "execute_result",
                        "data": content["data"],
                    })
                case "display_data":
                    result.outputs.append({
                        "type": "display_data",
                        "data": content["data"],
                    })
                case "error":
                    result.status = "error"
                    result.error_name = content["ename"]
                    result.error_value = content["evalue"]
                    result.traceback = content.get("traceback", [])
                case "execute_reply":
                    result.status = content["status"]
                    if content["status"] == "error":
                        result.error_name = content.get("ename", "")
                        result.error_value = content.get("evalue", "")
                        result.traceback = content.get("traceback", [])
                    return result
                case "status":
                    pass

        raise KernelError("Execution timed out")

    def complete(self, code: str, cursor_pos: int) -> list[str]:
        """Request tab completions from the kernel."""
        if not self._ws:
            return []

        header = self._make_header("complete_request")
        msg = {
            "header": header,
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "cursor_pos": cursor_pos,
            },
            "channel": "shell",
        }
        msg_id = header["msg_id"]
        self._ws.send(json.dumps(msg))

        deadline = time.time() + 5.0
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._ws.settimeout(remaining)
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                break

            if isinstance(raw, bytes):
                continue
            try:
                reply = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            parent_msg_id = reply.get("parent_header", {}).get("msg_id")
            if parent_msg_id != msg_id:
                continue

            if reply["header"]["msg_type"] == "complete_reply":
                return reply["content"].get("matches", [])

        return []

    def repl(self, display) -> None:
        display.print_info("Connected to remote kernel. Type 'exit()' or Ctrl+D to quit.")
        execution_count = 1

        while True:
            try:
                code = input(f"In [{execution_count}]: ")
            except (EOFError, KeyboardInterrupt):
                display.print_info("\nDisconnecting from kernel.")
                break

            if not code.strip():
                continue
            if code.strip() == "exit()":
                break

            # Multi-line input: keep reading if line ends with ':'
            while code.rstrip().endswith(":") or code.rstrip().endswith("\\"):
                try:
                    continuation = input("   ...: ")
                    code += "\n" + continuation
                except (EOFError, KeyboardInterrupt):
                    break

            try:
                result = self.execute(code)
                display.print_execution_result(result, execution_count)
            except KernelError as e:
                display.print_error(str(e))

            execution_count += 1
