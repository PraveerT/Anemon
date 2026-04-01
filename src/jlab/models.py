from dataclasses import dataclass, field
from enum import Enum


class ContentType(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    NOTEBOOK = "notebook"


@dataclass
class ContentItem:
    name: str
    path: str
    type: ContentType
    size: int | None = None
    last_modified: str = ""
    writable: bool = True
    content: str | list | dict | None = None
    format: str | None = None
    mimetype: str | None = None


@dataclass
class KernelInfo:
    id: str
    name: str
    last_activity: str
    execution_state: str
    connections: int


@dataclass
class SessionInfo:
    id: str
    path: str
    name: str
    type: str
    kernel: KernelInfo


@dataclass
class ServerStatus:
    started: str
    last_activity: str
    connections: int
    kernels: int


@dataclass
class ExecutionResult:
    status: str  # "ok" or "error"
    execution_count: int = 0
    outputs: list[dict] = field(default_factory=list)
    error_name: str = ""
    error_value: str = ""
    traceback: list[str] = field(default_factory=list)
