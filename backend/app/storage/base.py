"""Abstract base class for document storage backends (Phase 9.1)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.models.documents import DocumentType, StoredDocument


class DocumentStorage(ABC):
    """Abstract interface for document file storage backends.

    Enables local filesystem storage for development, while allowing seamless
    future migration to cloud object storage (e.g. AWS S3, Google Cloud Storage).
    """

    @abstractmethod
    def save(
        self,
        content: bytes,
        original_filename: str,
        ticker: str,
        document_type: DocumentType,
        mime_type: str = "application/pdf",
    ) -> StoredDocument:
        """Save file content into storage and return document metadata.

        Args:
            content: Raw binary content of the file.
            original_filename: Client-submitted filename (sanitized before storage).
            ticker: Normalized ticker symbol.
            document_type: Category of the financial document.
            mime_type: MIME content type.

        Returns:
            StoredDocument: Metadata describing the stored file.

        Raises:
            StoragePathTraversalError: If filename or path attempts path traversal.
            StorageIOError: If writing to storage fails.
        """
        pass

    @abstractmethod
    def get_bytes(self, storage_path: str) -> bytes:
        """Retrieve the raw binary content of a stored document.

        Args:
            storage_path: Relative storage reference key returned by save().

        Returns:
            bytes: Stored file contents.

        Raises:
            StorageNotFoundError: If document does not exist.
            StoragePathTraversalError: If path attempts path traversal.
            StorageIOError: If reading fails.
        """
        pass

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Check if a stored document exists.

        Args:
            storage_path: Relative storage reference key returned by save().

        Returns:
            bool: True if exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, storage_path: str) -> bool:
        """Delete a stored document if it exists.

        Args:
            storage_path: Relative storage reference key returned by save().

        Returns:
            bool: True if deleted, False if did not exist.
        """
        pass

    @abstractmethod
    def get_filesystem_path(self, storage_path: str) -> Optional[Path]:
        """Return physical filesystem Path if backend is local, or None if remote."""
        pass
