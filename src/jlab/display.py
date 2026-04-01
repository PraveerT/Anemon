from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from jlab.models import ContentItem, ContentType, KernelInfo, ServerStatus, ExecutionResult


def _format_size(size: int | None) -> str:
    if size is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class DisplayFormatter:
    def __init__(self):
        self.console = Console()

    def print_status(self, status: ServerStatus) -> None:
        table = Table(show_header=False, box=None)
        table.add_row("Started:", status.started)
        table.add_row("Last Activity:", status.last_activity)
        table.add_row("Connections:", str(status.connections))
        table.add_row("Running Kernels:", str(status.kernels))
        self.console.print(Panel(table, title="[bold green]Server Status"))

    def print_contents(self, items: list[ContentItem]) -> None:
        table = Table(title="Files")
        table.add_column("Type", style="cyan", width=4)
        table.add_column("Name", style="white")
        table.add_column("Size", justify="right")
        table.add_column("Modified")

        icons = {
            ContentType.DIRECTORY: "[blue]DIR",
            ContentType.FILE: "   ",
            ContentType.NOTEBOOK: "[green]NB",
        }

        for item in sorted(items, key=lambda x: (x.type != ContentType.DIRECTORY, x.name)):
            table.add_row(
                icons.get(item.type, "   "),
                item.name,
                _format_size(item.size),
                item.last_modified[:19] if item.last_modified else "",
            )
        self.console.print(table)

    def print_file_content(self, item: ContentItem) -> None:
        if item.content is None:
            self.print_error("No content available")
            return

        if item.name.endswith(".py"):
            syntax = Syntax(str(item.content), "python", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        elif item.name.endswith(".ipynb") and isinstance(item.content, dict):
            self._render_notebook(item.content)
        else:
            self.console.print(str(item.content))

    def _render_notebook(self, nb: dict) -> None:
        for i, cell in enumerate(nb.get("cells", []), 1):
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)
            cell_type = cell["cell_type"]

            if cell_type == "code":
                self.console.rule(f"[bold]In [{i}]")
                syntax = Syntax(source, "python", theme="monokai", line_numbers=True)
                self.console.print(syntax)
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "stream":
                        self.console.print("".join(output.get("text", [])), end="")
                    elif output.get("output_type") in ("execute_result", "display_data"):
                        data = output.get("data", {})
                        self.console.print(data.get("text/plain", ""))
            elif cell_type == "markdown":
                self.console.rule(f"[dim]Markdown [{i}]")
                self.console.print(source)

    def print_kernels(self, kernels: list[KernelInfo]) -> None:
        if not kernels:
            self.print_info("No running kernels")
            return
        table = Table(title="Running Kernels")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("State", style="green")
        table.add_column("Connections", justify="right")
        table.add_column("Last Activity")
        for k in kernels:
            state_style = "green" if k.execution_state == "idle" else "yellow"
            table.add_row(
                k.id[:12] + "...",
                k.name,
                Text(k.execution_state, style=state_style),
                str(k.connections),
                k.last_activity[:19],
            )
        self.console.print(table)

    def print_execution_result(self, result: ExecutionResult, count: int) -> None:
        for output in result.outputs:
            match output["type"]:
                case "stream":
                    style = "red" if output["name"] == "stderr" else ""
                    self.console.print(output["text"], style=style, end="")
                case "execute_result":
                    data = output["data"]
                    text = data.get("text/plain", "")
                    self.console.print(f"Out[{count}]: {text}", style="bold")
                case "display_data":
                    data = output["data"]
                    self.console.print(data.get("text/plain", data.get("text/html", "")))

        if result.status == "error":
            for line in result.traceback:
                self.console.print(line)

    def print_success(self, message: str) -> None:
        self.console.print(f"[bold green]OK[/] {message}")

    def print_error(self, message: str) -> None:
        self.console.print(f"[bold red]Error:[/] {message}")

    def print_info(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/]")

    def print_cell_header(self, index: int, total: int, source: str) -> None:
        preview = source.split("\n")[0][:60]
        self.console.rule(f"[bold]Cell {index}/{total}[/]: {preview}")
