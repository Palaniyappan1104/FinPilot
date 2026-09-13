"""Local filesystem implementation of DocumentStorage (Phase 9.1)."""

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from app.core.logging import get_logger
from app.models.documents import DocumentType, StoredDocument
from app.storage.base import DocumentStorage
from app.storage.exceptions import (
    StorageIOError,
    StorageNotFoundError,
    StoragePathTraversalError,
)

logger = get_logger("app.storage.local")


class LocalDocumentStorage(DocumentStorage):
    """Stores uploaded research documents on the local filesystem.

    Enforces strict security boundaries:
    - Path traversal prevention via canonical path resolution against storage root.
    - Sanitization of client-submitted filenames.
    - Collision-resistant document IDs and filenames using UUID4 hex strings.
    - Organized subdirectories by ticker symbol.
    """

    def __init__(self, base_directory: Union[str, Path]) -> None:
        """Initialize LocalDocumentStorage and ensure the root directory exists.

        Args:
            base_directory: Root directory path for document storage.
        """
        self.base_directory = Path(base_directory).resolve()
        try:
            self.base_directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StorageIOError(
                f"Failed to initialize storage directory '{self.base_directory}': {e}"
            ) from e

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize client-provided filename to prevent path injection.

        Extracts the basename, removes dangerous characters, and ensures a safe name.
        """
        # Strip directory path separators and null bytes
        base = os.path.basename(filename.replace("\\", "/")).strip("\x00 \t\n\r")
        # Keep only alphanumeric, dots, dashes, and underscores
        safe = re.sub(r"[^\w\-.]", "_", base)
        # Collapse multiple underscores/dots
        safe = re.sub(r"_+", "_", safe)
        if not safe or safe in {".", ".."}:
            return "document.pdf"
        return safe

    def _sanitize_ticker(self, ticker: str) -> str:
        """Sanitize ticker symbol for safe directory naming."""
        clean = re.sub(r"[^\w\-.]", "", ticker.strip().upper())
        return clean or "UNKNOWN"

    def _resolve_safe_path(self, storage_path: str) -> Path:
        """Resolve a relative storage reference key and enforce root boundary."""
        if "\x00" in storage_path:
            raise StoragePathTraversalError("Storage path contains invalid null bytes.")

        # Normalize separators
        clean_rel = storage_path.replace("\\", "/").lstrip("/")
        target = (self.base_directory / clean_rel).resolve()

        try:
            target.relative_to(self.base_directory)
        except ValueError as e:
            raise StoragePathTraversalError(
                f"Path traversal detected: '{storage_path}' resolves outside "
                "storage root."
            ) from e

        return target

    def save(
        self,
        content: bytes,
        original_filename: str,
        ticker: str,
        document_type: DocumentType,
        mime_type: str = "application/pdf",
    ) -> StoredDocument:
        """Save file content to local disk and return StoredDocument metadata."""
        clean_ticker = self._sanitize_ticker(ticker)
        safe_orig_name = self._sanitize_filename(original_filename)

        # Generate unique server-side identifier
        doc_id = f"doc_{uuid.uuid4().hex[:16]}"
        stored_filename = f"{doc_id}_{safe_orig_name}"
        rel_storage_path = f"{clean_ticker}/{stored_filename}"

        target_file_path = self._resolve_safe_path(rel_storage_path)

        try:
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            target_file_path.write_bytes(content)
        except OSError as e:
            logger.error(
                "I/O error saving document '%s' to '%s': %s",
                original_filename,
                target_file_path,
                e,
            )
            raise StorageIOError(f"Failed to write document to storage: {e}") from e

        sha256 = hashlib.sha256(content).hexdigest()

        return StoredDocument(
            document_id=doc_id,
            ticker=clean_ticker,
            document_type=document_type,
            original_filename=safe_orig_name,
            stored_filename=stored_filename,
            storage_path=rel_storage_path,
            file_size_bytes=len(content),
            sha256_checksum=sha256,
            mime_type=mime_type,
            uploaded_at=datetime.now(timezone.utc),
        )

    def get_bytes(self, storage_path: str) -> bytes:
        """Read and return binary content of stored document."""
        target_path = self._resolve_safe_path(storage_path)
        if not target_path.is_file():
            raise StorageNotFoundError(
                f"Document not found in storage: '{storage_path}'."
            )

        try:
            return target_path.read_bytes()
        except OSError as e:
            raise StorageIOError(
                f"Failed to read document from storage '{storage_path}': {e}"
            ) from e

    def exists(self, storage_path: str) -> bool:
        """Check if stored document exists."""
        try:
            target_path = self._resolve_safe_path(storage_path)
            return target_path.is_file()
        except StoragePathTraversalError:
            return False

    def delete(self, storage_path: str) -> bool:
        """Delete stored document from local filesystem."""
        try:
            target_path = self._resolve_safe_path(storage_path)
            if target_path.is_file():
                target_path.unlink()
                return True
            return False
        except (OSError, StoragePathTraversalError) as e:
            logger.warning("Failed to delete document '%s': %s", storage_path, e)
            return False

    def get_filesystem_path(self, storage_path: str) -> Optional[Path]:
        """Return physical local filesystem Path."""
        return self._resolve_safe_path(storage_path)
