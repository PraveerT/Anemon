from dataclasses import dataclass, field

from jlab.client import JupyterClient
from jlab.display import DisplayFormatter
from jlab.kernel import KernelConnection
from jlab.models import ExecutionResult


@dataclass
class NotebookCell:
    cell_type: str
    source: str
    execution_count: int | None = None
    outputs: list = field(default_factory=list)


def parse_notebook(content: dict) -> list[NotebookCell]:
    cells = []
    for cell_data in content.get("cells", []):
        source = cell_data.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        cells.append(NotebookCell(
            cell_type=cell_data["cell_type"],
            source=source,
            execution_count=cell_data.get("execution_count"),
        ))
    return cells


def run_notebook(
    client: JupyterClient,
    kernel_conn: KernelConnection,
    notebook_path: str,
    display: DisplayFormatter,
) -> list[ExecutionResult]:
    item = client.get_contents(notebook_path)
    if item.type.value != "notebook":
        raise ValueError(f"{notebook_path} is not a notebook")

    cells = parse_notebook(item.content)
    code_cells = [c for c in cells if c.cell_type == "code" and c.source.strip()]

    display.print_info(f"Running {len(code_cells)} code cells from {notebook_path}")
    results = []

    for i, cell in enumerate(code_cells, 1):
        display.print_cell_header(i, len(code_cells), cell.source)
        result = kernel_conn.execute(cell.source)
        display.print_execution_result(result, i)
        results.append(result)

        if result.status == "error":
            display.print_error(
                f"Cell {i} failed: {result.error_name}: {result.error_value}"
            )
            display.print_error("Stopping notebook execution.")
            break

    return results
