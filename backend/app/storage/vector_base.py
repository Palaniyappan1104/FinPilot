"""Vector Store abstract interface and collection naming strategies (Phase 9.7).

Defines:
- VectorStore: Abstract interface for vector database engines.
- build_document_collection_name: Keyed by company/document (finpilot_{ticker}_{doc}).
- build_company_collection_name: Keyed by company (finpilot_company_{ticker}).
- validate_collection_name: Enforces ChromaDB/vector store naming rules.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.vector_store import VectorRecord, VectorStoreInsertionResult
from app.storage.vector_exceptions import (
    InvalidCollectionNameError,
)

# ChromaDB collection naming constraints:
# 3-512 chars, [a-zA-Z0-9._-], starts and ends with [a-zA-Z0-9], no consecutive periods
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")


def sanitize_collection_name_part(part: str) -> str:
    """Sanitize an individual name component (e.g. ticker or document_id).

    Replaces unsupported characters with underscores, collapses consecutive
    separators, and strips leading/trailing non-alphanumerics.
    """
    if not part:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", part.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_-")


def validate_collection_name(name: str) -> str:
    """Validate that a collection name satisfies vector store naming constraints.

    Args:
        name: Proposed collection name string.

    Returns:
        str: Validated collection name.

    Raises:
        InvalidCollectionNameError: If name violates length or format rules.
    """
    if not name or not isinstance(name, str):
        raise InvalidCollectionNameError(
            "Collection name must be a non-empty string.", collection=str(name)
        )

    name = name.strip()

    if len(name) < 3 or len(name) > 512:
        raise InvalidCollectionNameError(
            f"Collection name '{name}' must be between 3 and 512 characters. "
            f"Got {len(name)} characters.",
            collection=name,
        )

    if ".." in name:
        raise InvalidCollectionNameError(
            f"Collection name '{name}' cannot contain consecutive periods ('..').",
            collection=name,
        )

    if not _COLLECTION_NAME_PATTERN.match(name):
        raise InvalidCollectionNameError(
            f"Collection name '{name}' is invalid. Names must contain only "
            "alphanumerics, underscores, hyphens, and periods, and must "
            "start and end with an alphanumeric character.",
            collection=name,
        )

    return name


def build_document_collection_name(ticker: str, document_id: str) -> str:
    """Build a deterministic collection name keyed by company and document.

    Format: finpilot_{ticker}_{document_id}

    Args:
        ticker: Uppercase or raw stock ticker symbol (e.g. 'AAPL').
        document_id: Unique document identifier (e.g. 'doc-annual-2024').

    Returns:
        str: Validated collection name (e.g. 'finpilot_aapl_doc-annual-2024').

    Raises:
        InvalidCollectionNameError: If ticker or document_id is missing or invalid.
    """
    clean_ticker = sanitize_collection_name_part(ticker)
    clean_doc = sanitize_collection_name_part(document_id)

    if not clean_ticker:
        raise InvalidCollectionNameError(
            f"Cannot build collection name: ticker '{ticker}' produced empty value."
        )
    if not clean_doc:
        raise InvalidCollectionNameError(
            f"Invalid document_id '{document_id}': produced empty value."
        )

    raw_name = f"finpilot_{clean_ticker}_{clean_doc}"
    return validate_collection_name(raw_name)


def build_company_collection_name(ticker: str) -> str:
    """Build a deterministic collection name keyed by company/ticker.

    Format: finpilot_company_{ticker}

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').

    Returns:
        str: Validated collection name (e.g. 'finpilot_company_aapl').

    Raises:
        InvalidCollectionNameError: If ticker is missing or produces an invalid name.
    """
    clean_ticker = sanitize_collection_name_part(ticker)
    if not clean_ticker:
        raise InvalidCollectionNameError(
            f"Invalid ticker '{ticker}': produced empty sanitized value."
        )
    raw_name = f"finpilot_company_{clean_ticker}"
    return validate_collection_name(raw_name)


class VectorStore(ABC):
    """Abstract interface defining the contract for vector database storage.

    Isolates vector database implementation details (such as ChromaDB) from the
    rest of the FinPilot system.

    Implementations must:
    - Never generate embeddings (only accept pre-computed vectors).
    - Provide idempotent upsert semantics (no duplicate records on re-runs).
    - Enforce domain validation on vectors, metadata, and collection names.
    - Raise typed VectorStoreError subclasses on all failures.
    """

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Return the vector store identifier (e.g. 'chroma')."""

    @abstractmethod
    def get_or_create_collection(self, collection_name: str) -> None:
        """Ensure a collection exists, creating it if necessary.

        Args:
            collection_name: Validated collection name.

        Raises:
            InvalidCollectionNameError: If collection name is invalid.
            VectorStoreError: If collection creation fails.
        """

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check whether a collection exists in the vector store.

        Args:
            collection_name: Collection name to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection and all its records.

        Args:
            collection_name: Name of collection to delete.

        Returns:
            bool: True if deleted, False if collection did not exist.

        Raises:
            InvalidCollectionNameError: If collection name is invalid.
            VectorStoreError: On database failure.
        """

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return a list of all existing collection names.

        Returns:
            List[str]: Alphabetical list of collection names.
        """

    @abstractmethod
    def upsert_records(
        self,
        collection_name: str,
        records: List[VectorRecord],
    ) -> VectorStoreInsertionResult:
        """Insert or update a batch of VectorRecords into the named collection.

        Idempotent: If a record with the same ID already exists, its vector,
        document text, and metadata are updated without creating duplicates.

        Args:
            collection_name: Target collection name.
            records: List of validated VectorRecord objects.

        Returns:
            VectorStoreInsertionResult: Summary of affected records.

        Raises:
            InvalidCollectionNameError: If collection name is invalid.
            DuplicateRecordIdError: If records contains duplicates within the batch.
            VectorValidationError: If any vector, document, or metadata is invalid.
            VectorDimensionMismatchError: If vectors have inconsistent dimensions.
            VectorStoreError: On underlying database failure.
        """

    @abstractmethod
    def count_records(self, collection_name: str) -> int:
        """Return the total number of records stored in a collection.

        Args:
            collection_name: Collection name.

        Returns:
            int: Number of records in collection.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            VectorStoreError: On database failure.
        """

    @abstractmethod
    def get_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> Optional[VectorRecord]:
        """Retrieve a single VectorRecord by its ID from a collection.

        Args:
            collection_name: Target collection name.
            record_id: Unique record/chunk ID.

        Returns:
            Optional[VectorRecord]: Retrieved record, or None if not found.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            VectorStoreError: On database failure.
        """

    @abstractmethod
    def get_records(
        self,
        collection_name: str,
        record_ids: List[str],
    ) -> List[VectorRecord]:
        """Retrieve multiple VectorRecords by their IDs from a collection.

        Args:
            collection_name: Target collection name.
            record_ids: List of chunk/record IDs to fetch.

        Returns:
            List[VectorRecord]: List of found VectorRecords.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            VectorStoreError: On database failure.
        """

    @abstractmethod
    def delete_records(
        self,
        collection_name: str,
        record_ids: List[str],
    ) -> int:
        """Delete specific records by their IDs from a collection.

        Args:
            collection_name: Target collection name.
            record_ids: IDs of records to remove.

        Returns:
            int: Number of records requested for deletion.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            VectorStoreError: On database failure.
        """
