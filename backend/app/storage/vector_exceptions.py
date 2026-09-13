"""Typed domain exceptions for Vector Store operations (Phase 9.7: ChromaDB Storage).

Defines:
- VectorStoreError: Base exception for all vector store operations.
- VectorStoreInitializationError: Storage setup, path creation, or engine init failed.
- VectorStoreConnectionError: Vector database engine connection failure.
- InvalidCollectionNameError: Collection name violates naming constraints.
- CollectionNotFoundError: Target collection does not exist.
- VectorValidationError: Vector, document text, or metadata failed domain validation.
- DuplicateRecordIdError: Batch contains duplicate record IDs.
- VectorDimensionMismatchError: Embedding dimension does not match expectation.
- EmptyBatchError: Attempted operation on an empty batch where disallowed.
"""

from typing import List, Optional


class VectorStoreError(Exception):
    """Base exception for all vector store failures."""

    def __init__(
        self,
        message: str,
        collection: Optional[str] = None,
        code: str = "VECTOR_STORE_ERROR",
    ) -> None:
        self.message = message
        self.collection = collection
        self.code = code
        prefix = f"[{collection}] " if collection else ""
        super().__init__(f"{prefix}[{code}] {message}")


class VectorStoreInitializationError(VectorStoreError):
    """Raised when vector store client or persistent directory cannot be initialized."""

    def __init__(self, message: str, path: Optional[str] = None) -> None:
        self.path = path
        super().__init__(
            message=message,
            code="INITIALIZATION_FAILED",
        )


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to the underlying vector store fails."""

    def __init__(self, message: str, collection: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            collection=collection,
            code="CONNECTION_ERROR",
        )


class InvalidCollectionNameError(VectorStoreError):
    """Raised when a collection name violates naming rules or constraints."""

    def __init__(self, message: str, collection: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            collection=collection,
            code="INVALID_COLLECTION_NAME",
        )


class CollectionNotFoundError(VectorStoreError):
    """Raised when an operation references a collection that does not exist."""

    def __init__(self, collection: str) -> None:
        super().__init__(
            message=f"Collection '{collection}' not found.",
            collection=collection,
            code="COLLECTION_NOT_FOUND",
        )


class VectorValidationError(VectorStoreError):
    """Raised when record data (vector, text, metadata) fails integrity validation."""

    def __init__(
        self,
        message: str,
        collection: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> None:
        self.record_id = record_id
        super().__init__(
            message=message,
            collection=collection,
            code="VALIDATION_ERROR",
        )


class DuplicateRecordIdError(VectorStoreError):
    """Raised when a batch contains duplicate record IDs."""

    def __init__(
        self,
        message: str,
        collection: Optional[str] = None,
        duplicate_ids: Optional[List[str]] = None,
    ) -> None:
        self.duplicate_ids = duplicate_ids or []
        super().__init__(
            message=message,
            collection=collection,
            code="DUPLICATE_RECORD_ID",
        )


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when embedding vector dimension does not match collection expectation."""

    def __init__(
        self,
        message: str,
        collection: Optional[str] = None,
        expected_dim: Optional[int] = None,
        actual_dim: Optional[int] = None,
    ) -> None:
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(
            message=message,
            collection=collection,
            code="DIMENSION_MISMATCH",
        )


class EmptyBatchError(VectorStoreError):
    """Raised when an empty batch of records is submitted where records are required."""

    def __init__(self, message: str, collection: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            collection=collection,
            code="EMPTY_BATCH",
        )
