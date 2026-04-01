class JlabError(Exception):
    """Base exception for all jlab errors."""


class ConnectionError(JlabError):
    """Cannot reach the JupyterLab server."""


class AuthenticationError(JlabError):
    """Token is invalid or missing."""


class NotFoundError(JlabError):
    """Requested resource (file, kernel, etc.) not found."""


class KernelError(JlabError):
    """Error during kernel communication."""


class ExecutionError(JlabError):
    """Code execution returned an error."""


class ConfigError(JlabError):
    """Configuration file missing or malformed."""
