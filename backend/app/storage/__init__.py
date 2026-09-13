"""Storage package for FinPilot Research Vault (Phase 9.1 and Phase 9.7)."""

from app.storage.base import DocumentStorage
from app.storage.chroma_vector_store import (
    ChromaVectorStore,
    get_vector_store,
)
from app.storage.exceptions import (
    StorageError,
    StorageIOError,
    StorageNotFoundError,
    StoragePathTraversalError,
)
from app.storage.local import LocalDocumentStorage
from app.storage.vector_base import (
    VectorStore,
    build_company_collection_name,
    build_document_collection_name,
    validate_collection_name,
)
from app.storage.vector_exceptions import (
    CollectionNotFoundError,
    DuplicateRecordIdError,
    EmptyBatchError,
    InvalidCollectionNameError,
    VectorDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorValidationError,
)

__all__ = [
    "ChromaVectorStore",
    "CollectionNotFoundError",
    "DocumentStorage",
    "DuplicateRecordIdError",
    "EmptyBatchError",
    "InvalidCollectionNameError",
    "LocalDocumentStorage",
    "StorageError",
    "StorageIOError",
    "StorageNotFoundError",
    "StoragePathTraversalError",
    "VectorDimensionMismatchError",
    "VectorStore",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "VectorStoreInitializationError",
    "VectorValidationError",
    "build_company_collection_name",
    "build_document_collection_name",
    "get_vector_store",
    "validate_collection_name",
]
