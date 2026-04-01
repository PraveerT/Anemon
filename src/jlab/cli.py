import functools
import sys
from pathlib import Path

import click

from jlab.config import JlabConfig, load_config, save_config
from jlab.client import JupyterClient
from jlab.display import DisplayFormatter
from jlab.exceptions import JlabError
from jlab.kernel import KernelConnection
from jlab.notebook import run_notebook

display = DisplayFormatter()


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
@click.argument("code")
@click.option("--kernel", "-k", default=None, help="Kernel name to use")
@handle_errors
def run(code: str, kernel: str | None):
    """Execute code on a remote kernel (one-shot)."""
    config = load_config()
    client = JupyterClient(config)
    kernel_name = kernel or config.default_kernel
    kernel_info = client.start_kernel(kernel_name)

    conn = KernelConnection(config, kernel_info.id)
    try:
        conn.connect()
        result = conn.execute(code)
        display.print_execution_result(result, 1)
        if result.status == "error":
            sys.exit(1)
    finally:
        conn.close()
        client.delete_kernel(kernel_info.id)


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


@main.command()
@handle_errors
def shell():
    """Open a remote shell via Python kernel (like SSH)."""
    config = load_config()
    client = JupyterClient(config)
    kernel_name = config.default_kernel
    display.print_info("Starting remote shell...")
    kernel_info = client.start_kernel(kernel_name)

    conn = KernelConnection(config, kernel_info.id)
    try:
        conn.connect()
        # Set up the kernel to run shell commands and get cwd
        conn.execute("import subprocess, os", timeout=10)
        result = conn.execute("os.getcwd()", timeout=10)
        cwd = ""
        for out in result.outputs:
            if out["type"] == "execute_result":
                cwd = out["data"].get("text/plain", "").strip("'\"")
        if not cwd:
            cwd = "~"

        display.print_info(f"Connected to remote shell. Type 'exit' or Ctrl+D to quit.\n")

        while True:
            try:
                cmd = input(f"\033[1;32mremote\033[0m:\033[1;34m{cwd}\033[0m$ ")
            except (EOFError, KeyboardInterrupt):
                display.print_info("\nDisconnecting.")
                break

            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd == "exit":
                break

            # Handle cd specially
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

            # Run shell command via subprocess
            code = (
                f"_r = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True, cwd={cwd!r})\n"
                f"if _r.stdout: print(_r.stdout, end='')\n"
                f"if _r.stderr: print(_r.stderr, end='')"
            )
            result = conn.execute(code, timeout=120)
            for out in result.outputs:
                if out["type"] == "stream":
                    sys.stdout.write(out["text"])
                    sys.stdout.flush()
            if result.status == "error" and result.traceback:
                for line in result.traceback:
                    display.console.print(line)

    finally:
        conn.close()
        try:
            client.delete_kernel(kernel_info.id)
        except Exception:
            pass


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
