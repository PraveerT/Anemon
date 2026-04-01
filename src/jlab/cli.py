import functools
import sys
from pathlib import Path

import click

from jlab.config import (
    JlabConfig, load_config, save_config,
    load_session, save_session, clear_session,
)
from jlab.client import JupyterClient
from jlab.display import DisplayFormatter
from jlab.exceptions import JlabError
from jlab.kernel import KernelConnection
from jlab.notebook import run_notebook
from jlab.terminal import TerminalConnection, _get_browser_cookies_and_headers

display = DisplayFormatter()


def _fix_remote_path(path: str) -> str:
    """Fix Git Bash path mangling. Git Bash converts /notebooks to
    C:/Users/.../Git/notebooks. Detect and strip the prefix."""
    if ":" in path and "/notebooks" in path:
        idx = path.find("/notebooks")
        return path[idx:]
    return path


def handle_errors(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except JlabError as e:
            display.print_error(str(e))
            sys.exit(1)
        except KeyboardInterrupt:
            display.print_info("\nInterrupted.")
            sys.exit(130)
    return wrapper


def get_client() -> JupyterClient:
    config = load_config()
    return JupyterClient(config)


def _get_session_conn(config, client):
    """Get a kernel connection from an active session, or None."""
    session = load_session()
    if not session:
        return None, None

    kernel_id = session["kernel_id"]
    cwd = session.get("cwd", "/notebooks")

    # Check if the kernel is still alive
    try:
        kernels = client.list_kernels()
        if not any(k.id == kernel_id for k in kernels):
            clear_session()
            return None, None
    except Exception:
        clear_session()
        return None, None

    conn = KernelConnection(config, kernel_id)
    conn.connect()
    return conn, cwd


@click.group()
@click.version_option(package_name="jlab")
def main():
    """jlab - Interact with a remote JupyterLab instance from the terminal."""
    pass


@main.command()
@click.argument("url")
@click.option("--token", "-t", default="", help="JupyterLab authentication token")
@click.option("--kernel", "-k", default="python3", help="Default kernel name")
@handle_errors
def connect(url: str, token: str, kernel: str):
    """Save connection configuration for a JupyterLab server."""
    config = JlabConfig(url=url.rstrip("/"), token=token, default_kernel=kernel)
    client = JupyterClient(config)
    status = client.status()
    save_config(config)
    display.print_success(f"Connected to {url}")
    display.print_status(status)


@main.command()
@handle_errors
def status():
    """Check server status."""
    client = get_client()
    st = client.status()
    display.print_status(st)


@main.command()
@click.argument("path", default="")
@handle_errors
def ls(path: str):
    """List files and directories on the server."""
    client = get_client()
    items = client.list_contents(path)
    display.print_contents(items)


@main.command()
@click.argument("path")
@handle_errors
def cat(path: str):
    """View file contents."""
    client = get_client()
    item = client.get_contents(path)
    display.print_file_content(item)


@main.command()
@click.argument("local", type=click.Path(exists=True))
@click.argument("remote")
@handle_errors
def upload(local: str, remote: str):
    """Upload a local file to the server."""
    client = get_client()
    result = client.upload_file(Path(local), remote)
    display.print_success(f"Uploaded {local} -> {result.path}")


@main.command()
@click.argument("remote")
@click.argument("local", required=False)
@handle_errors
def download(remote: str, local: str | None):
    """Download a file from the server."""
    client = get_client()
    content, fmt = client.download_file(remote)
    local_path = Path(local) if local else Path(remote.split("/")[-1])
    if isinstance(content, bytes):
        local_path.write_bytes(content)
    else:
        local_path.write_text(content)
    display.print_success(f"Downloaded {remote} -> {local_path}")


@main.command()
@click.argument("path")
@click.confirmation_option(prompt="Are you sure you want to delete this file?")
@handle_errors
def rm(path: str):
    """Delete a file on the server."""
    client = get_client()
    client.delete(path)
    display.print_success(f"Deleted {path}")


@main.command()
@handle_errors
def kernels():
    """List running kernels."""
    client = get_client()
    kernel_list = client.list_kernels()
    display.print_kernels(kernel_list)


@main.command()
@click.argument("code_snippets", nargs=-1, required=True, metavar="CODE")
@click.option("--kernel", "-k", default=None, help="Kernel name to use")
@handle_errors
def run(code_snippets: tuple[str, ...], kernel: str | None):
    """Execute Python code on a remote kernel.

    Accepts multiple code snippets to execute in a single connection:

        jlab run "import torch" "print(torch.cuda.is_available())" "print(torch.__version__)"
    """
    config = load_config()
    client = JupyterClient(config)

    # Try to use active session
    conn, _ = _get_session_conn(config, client)
    owns_kernel = False

    if not conn:
        kernel_name = kernel or config.default_kernel
        kernel_info = client.start_kernel(kernel_name)
        conn = KernelConnection(config, kernel_info.id)
        conn.connect()
        owns_kernel = True

    try:
        has_error = False
        for i, code in enumerate(code_snippets):
            if len(code_snippets) > 1:
                display.console.print(f"[bold cyan]>>> {code}[/bold cyan]")
            result = conn.execute(code)
            display.print_execution_result(result, i + 1)
            if result.status == "error":
                has_error = True
        if has_error:
            sys.exit(1)
    finally:
        conn.close()
        if owns_kernel:
            client.delete_kernel(kernel_info.id)


@main.command(name="exec")
@click.argument("commands", nargs=-1, required=True)
@click.option("--cwd", default=None, help="Working directory on remote")
@handle_errors
def exec_cmd(commands: tuple[str, ...], cwd: str | None):
    """Run shell command(s) on the remote machine.

    Accepts multiple commands to execute in a single connection:

        jlab exec "ls /notebooks" "cat README.md" "wc -l *.py"
    """
    if cwd:
        cwd = _fix_remote_path(cwd)
    config = load_config()
    client = JupyterClient(config)

    # Try to use active session
    conn, session_cwd = _get_session_conn(config, client)
    owns_kernel = False

    if not conn:
        kernel_info = client.start_kernel(config.default_kernel)
        conn = KernelConnection(config, kernel_info.id)
        conn.connect()
        conn.execute("import subprocess, os", timeout=10)
        owns_kernel = True
        session_cwd = None

    try:
        effective_cwd = cwd or session_cwd
        cwd_arg = f", cwd={effective_cwd!r}" if effective_cwd else ""
        has_error = False

        for i, command in enumerate(commands):
            # Print header when running multiple commands
            if len(commands) > 1:
                display.console.print(f"[bold cyan]$ {command}[/bold cyan]")

            code = (
                f"import subprocess as _sp\n"
                f"_r = _sp.run({command!r}, shell=True, capture_output=True, text=True{cwd_arg})\n"
                f"if _r.stdout: print(_r.stdout, end='')\n"
                f"if _r.stderr: print(_r.stderr, end='')"
            )
            result = conn.execute_streaming(code, timeout=600)
            if result.status == "error" and result.traceback:
                for line in result.traceback:
                    display.console.print(line)
                has_error = True

            # Separator between commands
            if len(commands) > 1 and i < len(commands) - 1:
                print()

        if has_error:
            sys.exit(1)
    finally:
        conn.close()
        if owns_kernel:
            client.delete_kernel(kernel_info.id)


# --- Session management ---

@main.group()
def session():
    """Manage persistent kernel sessions."""
    pass


@session.command("start")
@click.option("--kernel", "-k", default=None, help="Kernel name to use")
@click.option("--cwd", default="/notebooks", help="Initial working directory")
@handle_errors
def session_start(kernel: str | None, cwd: str):
    """Start a persistent kernel session."""
    cwd = _fix_remote_path(cwd)
    config = load_config()
    client = JupyterClient(config)

    # Check if a session already exists
    existing = load_session()
    if existing:
        try:
            kernels = client.list_kernels()
            if any(k.id == existing["kernel_id"] for k in kernels):
                display.print_info(f"Session already active (kernel: {existing['kernel_id'][:12]}...)")
                return
        except Exception:
            pass
        clear_session()

    kernel_name = kernel or config.default_kernel
    kernel_info = client.start_kernel(kernel_name)

    # Initialize the kernel
    conn = KernelConnection(config, kernel_info.id)
    conn.connect()
    conn.execute("import subprocess, os", timeout=10)
    if cwd:
        conn.execute(f"os.chdir({cwd!r})", timeout=10)
    conn.close()

    save_session(kernel_info.id, cwd)
    display.print_success(f"Session started (kernel: {kernel_info.id[:12]}..., cwd: {cwd})")


@session.command("stop")
@handle_errors
def session_stop():
    """Stop the persistent kernel session."""
    config = load_config()
    client = JupyterClient(config)

    session_data = load_session()
    if not session_data:
        display.print_info("No active session")
        return

    try:
        client.delete_kernel(session_data["kernel_id"])
    except Exception:
        pass

    clear_session()
    display.print_success("Session stopped")


@session.command("status")
@handle_errors
def session_status():
    """Check if a session is active."""
    config = load_config()
    client = JupyterClient(config)

    session_data = load_session()
    if not session_data:
        display.print_info("No active session. Run 'jlab session start' to create one.")
        return

    kernel_id = session_data["kernel_id"]
    try:
        kernels = client.list_kernels()
        alive = any(k.id == kernel_id for k in kernels)
    except Exception:
        alive = False

    if alive:
        display.print_success(f"Session active (kernel: {kernel_id[:12]}..., cwd: {session_data.get('cwd', '?')})")
    else:
        clear_session()
        display.print_info("Session expired (kernel no longer running). Run 'jlab session start' to create a new one.")


@session.command("cd")
@click.argument("path")
@handle_errors
def session_cd(path: str):
    """Change the session working directory."""
    path = _fix_remote_path(path)
    config = load_config()
    client = JupyterClient(config)

    conn, _ = _get_session_conn(config, client)
    if not conn:
        display.print_error("No active session. Run 'jlab session start' first.")
        sys.exit(1)

    try:
        code = f"os.chdir(os.path.expanduser({path!r}))\nprint(os.getcwd())"
        result = conn.execute(code, timeout=10)
        new_cwd = ""
        for out in result.outputs:
            if out["type"] == "stream" and out["name"] == "stdout":
                new_cwd = out["text"].strip()
        if new_cwd:
            session_data = load_session()
            save_session(session_data["kernel_id"], new_cwd)
            display.print_success(f"cwd: {new_cwd}")
        if result.status == "error" and result.traceback:
            for line in result.traceback:
                display.console.print(line)
    finally:
        conn.close()


# --- Find command ---

@main.command()
@click.argument("pattern")
@click.option("--path", "-p", default="/notebooks", help="Directory to search in")
@handle_errors
def find(pattern: str, path: str):
    """Find files on the remote machine by name pattern."""
    config = load_config()
    client = JupyterClient(config)

    conn, _ = _get_session_conn(config, client)
    owns_kernel = False

    if not conn:
        kernel_info = client.start_kernel(config.default_kernel)
        conn = KernelConnection(config, kernel_info.id)
        conn.connect()
        owns_kernel = True

    try:
        code = (
            f"import subprocess as _sp\n"
            f"_r = _sp.run(['find', {path!r}, '-name', {pattern!r}, '-type', 'f'], "
            f"capture_output=True, text=True, timeout=30)\n"
            f"if _r.stdout: print(_r.stdout, end='')\n"
            f"if _r.stderr: print(_r.stderr, end='')"
        )
        result = conn.execute_streaming(code, timeout=60)
        if result.status == "error" and result.traceback:
            for line in result.traceback:
                display.console.print(line)
    finally:
        conn.close()
        if owns_kernel:
            client.delete_kernel(kernel_info.id)


# --- Interactive shell ---

@main.command()
@click.option("--kernel", "-k", default=None, help="Kernel name to use")
@handle_errors
def repl(kernel: str | None):
    """Start an interactive REPL connected to a remote kernel."""
    config = load_config()
    client = JupyterClient(config)
    kernel_name = kernel or config.default_kernel
    kernel_info = client.start_kernel(kernel_name)

    conn = KernelConnection(config, kernel_info.id)
    try:
        conn.connect()
        conn.repl(display)
    finally:
        conn.close()
        client.delete_kernel(kernel_info.id)


def _shell_pty(config, client, display):
    """Real PTY shell via JupyterLab terminals WebSocket."""
    try:
        resp = client._request("GET", "terminals")
        for t in resp.json():
            client.delete_terminal(t["name"])
    except Exception:
        pass

    term = client.create_terminal()
    term_name = term["name"]
    display.print_info(f"Connected to remote terminal (pty:{term_name}). Press Ctrl+D to disconnect.\n")

    conn = TerminalConnection(config, term_name)
    try:
        conn.connect()
        conn.interactive()
    finally:
        conn.close()
        try:
            client.delete_terminal(term_name)
        except Exception:
            pass


def _shell_kernel(config, client, display):
    """Fallback shell via Python kernel + subprocess."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI

    kernel_name = config.default_kernel
    kernel_info = client.start_kernel(kernel_name)

    conn = KernelConnection(config, kernel_info.id)
    try:
        conn.connect()
        conn.execute("import subprocess, os, glob", timeout=10)
        result = conn.execute("os.getcwd()", timeout=10)
        cwd = ""
        for out in result.outputs:
            if out["type"] == "execute_result":
                cwd = out["data"].get("text/plain", "").strip("'\"")
        if not cwd:
            cwd = "~"

        class RemoteCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                parts = text.split()
                word = parts[-1] if parts and not text.endswith(" ") else ""

                code = (
                    f"import os as _os\n"
                    f"_w = {word!r}\n"
                    f"_dir = _os.path.dirname(_w) or '.'\n"
                    f"_pre = _os.path.basename(_w)\n"
                    f"_pfx = _os.path.dirname(_w)\n"
                    f"try:\n"
                    f"  _items = _os.listdir(_os.path.join({cwd!r}, _dir))\n"
                    f"  for _i in sorted(_items):\n"
                    f"    if _i.startswith(_pre):\n"
                    f"      _full = (_pfx + '/' + _i) if _pfx else _i\n"
                    f"      if _os.path.isdir(_os.path.join({cwd!r}, _dir, _i)): _full += '/'\n"
                    f"      print(_full)\n"
                    f"except: pass"
                )
                try:
                    r = conn.execute(code, timeout=5)
                    for out in r.outputs:
                        if out["type"] == "stream" and out["name"] == "stdout":
                            for match in out["text"].strip().split("\n"):
                                match = match.strip()
                                if match:
                                    yield Completion(match, start_position=-len(word))
                except Exception:
                    pass

        session = PromptSession(completer=RemoteCompleter(), complete_while_typing=False)
        display.print_info(f"Connected to remote shell (kernel mode). Type 'exit' or Ctrl+D to quit.\n")

        while True:
            try:
                prompt_text = ANSI(f"\033[1;32mremote\033[0m:\033[1;34m{cwd}\033[0m$ ")
                cmd = session.prompt(prompt_text)
            except (EOFError, KeyboardInterrupt):
                display.print_info("\nDisconnecting.")
                break

            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd == "exit":
                break

            if cmd == "clear":
                click.clear()
                continue

            if cmd == "cd" or cmd.startswith("cd "):
                path = cmd[3:].strip() or "~"
                code = (
                    f"os.chdir(os.path.expanduser({path!r}))\n"
                    f"print(os.getcwd())"
                )
                result = conn.execute(code, timeout=15)
                for out in result.outputs:
                    if out["type"] == "stream" and out["name"] == "stdout":
                        cwd = out["text"].strip()
                if result.status == "error" and result.traceback:
                    for line in result.traceback:
                        display.console.print(line)
                continue

            code = (
                f"import subprocess, sys\n"
                f"_p = subprocess.Popen({cmd!r}, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd={cwd!r}, bufsize=1)\n"
                f"for _line in _p.stdout:\n"
                f"    print(_line, end='', flush=True)\n"
                f"_p.wait()"
            )
            result = conn.execute_streaming(code, timeout=600)
            if result.status == "error" and result.traceback:
                for line in result.traceback:
                    display.console.print(line)

    finally:
        conn.close()
        try:
            client.delete_kernel(kernel_info.id)
        except Exception:
            pass


@main.command()
@click.option("--mode", type=click.Choice(["auto", "pty", "kernel"]), default="auto",
              help="Shell mode: pty (real terminal), kernel (subprocess), auto (try pty first)")
@handle_errors
def shell(mode: str):
    """Open an interactive remote shell (like SSH)."""
    config = load_config()
    client = JupyterClient(config)

    if mode == "kernel":
        _shell_kernel(config, client, display)
        return

    if mode == "pty":
        _shell_pty(config, client, display)
        return

    # Auto mode: try PTY first, fall back to kernel
    try:
        display.print_info("Connecting to remote terminal...")
        _shell_pty(config, client, display)
    except Exception as e:
        display.print_info(f"PTY terminal unavailable ({e}), falling back to kernel mode...")
        _shell_kernel(config, client, display)


# --- Notebook subgroup ---

@main.group()
def nb():
    """Notebook operations."""
    pass


@nb.command("run")
@click.argument("notebook")
@click.option("--kernel", "-k", default=None, help="Kernel name to use")
@handle_errors
def nb_run(notebook: str, kernel: str | None):
    """Run all cells of a notebook on the server."""
    config = load_config()
    client = JupyterClient(config)
    kernel_name = kernel or config.default_kernel
    kernel_info = client.start_kernel(kernel_name)

    conn = KernelConnection(config, kernel_info.id)
    try:
        conn.connect()
        results = run_notebook(client, conn, notebook, display)
        passed = sum(1 for r in results if r.status == "ok")
        failed = sum(1 for r in results if r.status == "error")
        display.print_info(f"\nResults: {passed} cells passed, {failed} cells failed")
        if failed > 0:
            sys.exit(1)
    finally:
        conn.close()
        client.delete_kernel(kernel_info.id)
