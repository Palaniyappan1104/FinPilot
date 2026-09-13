"""Storage exceptions for FinPilot Research Vault (Phase 9.1)."""


class StorageError(Exception):
    """Base exception for all storage operations."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class StoragePathTraversalError(StorageError):
    """Raised when an operation attempts directory traversal outside storage root."""


class StorageIOError(StorageError):
    """Raised when an I/O operation fails while reading or writing storage."""


class StorageNotFoundError(StorageError):
    """Raised when a requested stored document or path does not exist."""
