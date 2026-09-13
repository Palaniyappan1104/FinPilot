"""ChromaDB implementation of the VectorStore interface (Phase 9.7: ChromaDB Storage).

Provides:
- ChromaVectorStore: Concrete implementation using ChromaDB's native collection APIs.
- get_vector_store: Factory function for instantiating the configured VectorStore.

Fully isolates ChromaDB from the rest of the application.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.vector_store import (
    VectorRecord,
    VectorSearchResult,
    VectorStoreInsertionResult,
)
from app.storage.vector_base import (
    VectorStore,
    validate_collection_name,
)
from app.storage.vector_exceptions import (
    CollectionNotFoundError,
    DuplicateRecordIdError,
    InvalidCollectionNameError,
    VectorDimensionMismatchError,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorValidationError,
)

logger = get_logger("app.storage.chroma_vector_store")


class ChromaVectorStore(VectorStore):
    """Concrete VectorStore implementation backed by ChromaDB.

    Supports:
    - PersistentClient (production with CHROMA_PERSIST_DIRECTORY)
    - EphemeralClient / injected client (testing and in-memory operations)
    - Idempotent upsert of chunk embeddings, documents, and flat primitive metadata.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialize the ChromaVectorStore.

        Args:
            settings: Application settings instance.
            client: Optional pre-configured ChromaDB client (e.g. EphemeralClient).
            persist_directory: Optional override for the ChromaDB persistence folder.

        Raises:
            VectorStoreInitializationError: If client or path cannot be initialized.
        """
        self._settings = settings or get_settings()

        if client is not None:
            self._client = client
            self._persist_directory = None
            logger.debug("ChromaVectorStore initialized with injected client.")
            return

        # Resolve persistence directory
        raw_path = (
            persist_directory
            or getattr(self._settings, "CHROMA_PERSIST_DIRECTORY", "./chroma_data")
            or "./chroma_data"
        )
        try:
            path = Path(raw_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            self._persist_directory = str(path)
        except Exception as exc:
            raise VectorStoreInitializationError(
                f"Failed to access ChromaDB directory '{raw_path}': {exc}",
                path=raw_path,
            ) from exc

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_directory)
            logger.info(
                "ChromaVectorStore initialized with persistent path: %s",
                self._persist_directory,
            )
        except Exception as exc:
            raise VectorStoreInitializationError(
                f"Failed to initialize ChromaDB PersistentClient: {exc}",
                path=self._persist_directory,
            ) from exc

    @property
    def store_name(self) -> str:
        return "chroma"

    @property
    def persist_directory(self) -> Optional[str]:
        return self._persist_directory

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, collection_name: str) -> Any:
        """Ensure the collection exists, creating it if needed."""
        validated_name = validate_collection_name(collection_name)
        try:
            return self._client.get_or_create_collection(name=validated_name)
        except InvalidCollectionNameError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get/create ChromaDB collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

    def has_collection(self, collection_name: str) -> bool:
        """Check whether the collection exists."""
        validated_name = validate_collection_name(collection_name)
        return validated_name in self.list_collections()

    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection. Returns True if deleted, False if not found."""
        validated_name = validate_collection_name(collection_name)
        if not self.has_collection(validated_name):
            return False
        try:
            self._client.delete_collection(name=validated_name)
            logger.info("Deleted ChromaDB collection: %s", validated_name)
            return True
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete ChromaDB collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

    def list_collections(self) -> List[str]:
        """List all collection names in the ChromaDB database."""
        try:
            cols = self._client.list_collections()
            names: List[str] = []
            for c in cols:
                if isinstance(c, str):
                    names.append(c)
                elif hasattr(c, "name"):
                    names.append(c.name)
            return sorted(names)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to list ChromaDB collections: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Record operations (Insert / Upsert)
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        collection_name: str,
        records: List[VectorRecord],
    ) -> VectorStoreInsertionResult:
        """Insert or update records in the specified collection.

        Validates all inputs, prevents duplicate IDs within the batch,
        ensures dimensional consistency, and calls ChromaDB's native upsert.
        """
        validated_name = validate_collection_name(collection_name)

        if not records:
            logger.debug(
                "upsert_records called with empty batch for collection '%s'. No-op.",
                validated_name,
            )
            return VectorStoreInsertionResult(
                collection_name=validated_name,
                record_ids=[],
                total_records=0,
            )

        self._validate_batch(validated_name, records)

        collection = self.get_or_create_collection(validated_name)

        ids = [r.id for r in records]
        embeddings = [r.embedding for r in records]
        documents = [r.document for r in records]
        metadatas = [r.metadata for r in records]

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(
                "Upserted %d records into ChromaDB collection '%s'.",
                len(records),
                validated_name,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert records into collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

        return VectorStoreInsertionResult(
            collection_name=validated_name,
            record_ids=ids,
            total_records=len(ids),
        )

    def count_records(self, collection_name: str) -> int:
        """Return the number of records in the specified collection."""
        validated_name = validate_collection_name(collection_name)
        if not self.has_collection(validated_name):
            raise CollectionNotFoundError(validated_name)
        try:
            col = self._client.get_collection(name=validated_name)
            return col.count()
        except CollectionNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count records in collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

    def get_record(
        self,
        collection_name: str,
        record_id: str,
    ) -> Optional[VectorRecord]:
        """Fetch a single record by ID."""
        records = self.get_records(collection_name, [record_id])
        return records[0] if records else None

    def get_records(
        self,
        collection_name: str,
        record_ids: List[str],
    ) -> List[VectorRecord]:
        """Fetch multiple records by ID."""
        validated_name = validate_collection_name(collection_name)
        if not self.has_collection(validated_name):
            raise CollectionNotFoundError(validated_name)
        if not record_ids:
            return []
        try:
            col = self._client.get_collection(name=validated_name)
            raw = col.get(
                ids=record_ids,
                include=["embeddings", "documents", "metadatas"],
            )
        except CollectionNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get records from collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

        found_ids = raw.get("ids", []) or []
        found_embeddings = raw.get("embeddings", [])
        found_documents = raw.get("documents", []) or []
        found_metadatas = raw.get("metadatas", []) or []

        results: List[VectorRecord] = []
        for i, rid in enumerate(found_ids):
            emb: List[float] = []
            if found_embeddings is not None and len(found_embeddings) > i:
                emb = list(found_embeddings[i])
            doc = found_documents[i] if len(found_documents) > i else ""
            meta = found_metadatas[i] if len(found_metadatas) > i else {}
            results.append(
                VectorRecord(
                    id=rid,
                    embedding=emb,
                    document=doc,
                    metadata=meta,
                )
            )
        return results

    def delete_records(
        self,
        collection_name: str,
        record_ids: List[str],
    ) -> int:
        """Delete records by ID from a collection."""
        validated_name = validate_collection_name(collection_name)
        if not self.has_collection(validated_name):
            raise CollectionNotFoundError(validated_name)
        if not record_ids:
            return 0
        try:
            col = self._client.get_collection(name=validated_name)
            col.delete(ids=record_ids)
            return len(record_ids)
        except CollectionNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete records from collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

    # ------------------------------------------------------------------
    # Similarity search (Phase 9.9)
    # ------------------------------------------------------------------

    def query_similarity(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> List[VectorSearchResult]:
        """Perform vector similarity search against a collection (Phase 9.9).

        Queries ChromaDB for the nearest stored chunks to query_embedding.
        Returns matches sorted by proximity with explicit distance semantics.

        Args:
            collection_name: Target collection name.
            query_embedding: Dense numeric query vector.
            n_results: Number of nearest matches to return (must be > 0).
            where: Optional metadata filter dictionary.

        Returns:
            List[VectorSearchResult]: Ranked matches with distances and provenance.

        Raises:
            InvalidCollectionNameError: If collection name is invalid.
            CollectionNotFoundError: If collection does not exist.
            VectorValidationError: If query vector or n_results is invalid.
            VectorDimensionMismatchError: If query vector dimension mismatches.
            VectorStoreError: On database or query failure.
        """
        validated_name = validate_collection_name(collection_name)

        if not self.has_collection(validated_name):
            raise CollectionNotFoundError(validated_name)

        if n_results <= 0:
            raise VectorValidationError(
                f"n_results must be greater than 0, got {n_results}.",
                collection=validated_name,
            )

        # Validate query embedding
        if not query_embedding or not isinstance(query_embedding, (list, tuple)):
            raise VectorValidationError(
                "Query embedding must be a non-empty list of floats.",
                collection=validated_name,
            )

        try:
            vec: List[float] = [float(x) for x in query_embedding]
        except (TypeError, ValueError) as exc:
            raise VectorValidationError(
                f"Query embedding contains non-numeric values: {exc}",
                collection=validated_name,
            ) from exc

        if not vec:
            raise VectorValidationError(
                "Query embedding must be a non-empty list of floats.",
                collection=validated_name,
            )

        non_finite = [x for x in vec if not math.isfinite(x)]
        if non_finite:
            raise VectorValidationError(
                f"Query embedding contains {len(non_finite)} non-finite value(s).",
                collection=validated_name,
            )

        try:
            col = self._client.get_collection(name=validated_name)
        except CollectionNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to access collection '{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

        count = col.count()
        if count == 0:
            logger.debug(
                "Collection '%s' is empty. Returning empty similarity results.",
                validated_name,
            )
            return []

        actual_n = min(n_results, count)

        # Determine distance metric from collection metadata if available
        metric_name = "cosine_distance"
        col_meta = getattr(col, "metadata", None) or {}
        hnsw_space = col_meta.get("hnsw:space") if isinstance(col_meta, dict) else None
        if hnsw_space == "l2":
            metric_name = "l2_distance"
        elif hnsw_space == "ip":
            metric_name = "inner_product"
        elif hnsw_space == "cosine":
            metric_name = "cosine_distance"

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [vec],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = col.query(**query_kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "dimension" in exc_str:
                raise VectorDimensionMismatchError(
                    f"Query vector dimension mismatch for collection "
                    f"'{validated_name}': {exc}",
                    collection=validated_name,
                ) from exc
            raise VectorStoreError(
                f"Failed to execute similarity query on collection "
                f"'{validated_name}': {exc}",
                collection=validated_name,
            ) from exc

        raw_ids_list = raw.get("ids") or []
        if not raw_ids_list or not raw_ids_list[0]:
            return []

        raw_ids = raw_ids_list[0]
        raw_distances_list = raw.get("distances") or []
        raw_documents_list = raw.get("documents") or []
        raw_metadatas_list = raw.get("metadatas") or []

        raw_distances = raw_distances_list[0] if raw_distances_list else []
        raw_documents = raw_documents_list[0] if raw_documents_list else []
        raw_metadatas = raw_metadatas_list[0] if raw_metadatas_list else []

        if len(raw_ids) != len(raw_distances) or len(raw_ids) != len(raw_documents):
            raise VectorStoreError(
                "Malformed response from ChromaDB: mismatched array lengths "
                f"(ids={len(raw_ids)}, distances={len(raw_distances)}, "
                f"documents={len(raw_documents)}).",
                collection=validated_name,
            )

        results: List[VectorSearchResult] = []
        for i, match_id in enumerate(raw_ids):
            meta = raw_metadatas[i] if raw_metadatas and i < len(raw_metadatas) else {}
            results.append(
                VectorSearchResult(
                    id=str(match_id),
                    distance=float(raw_distances[i]),
                    document=str(raw_documents[i]),
                    metadata=meta if isinstance(meta, dict) else {},
                    distance_metric=metric_name,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Batch validation helper
    # ------------------------------------------------------------------

    def _validate_batch(
        self, collection_name: str, records: List[VectorRecord]
    ) -> None:
        """Validate batch integrity: non-empty, uniqueness, dimensions, primitives."""
        # 1. Check duplicate IDs within the batch
        seen_ids = set()
        duplicates = []
        for r in records:
            if r.id in seen_ids:
                duplicates.append(r.id)
            seen_ids.add(r.id)
        if duplicates:
            raise DuplicateRecordIdError(
                f"Batch contains duplicate record ID(s): {duplicates}",
                collection=collection_name,
                duplicate_ids=duplicates,
            )

        # 2. Check dimension consistency across batch
        dims = {len(r.embedding) for r in records}
        if len(dims) > 1:
            raise VectorDimensionMismatchError(
                f"Inconsistent vector dimensions across batch: {dims}",
                collection=collection_name,
            )

        # 3. Validate each record
        for r in records:
            if not r.id or not r.id.strip():
                raise VectorValidationError(
                    "Record has empty or blank ID.",
                    collection=collection_name,
                    record_id=r.id,
                )
            if not r.document or not r.document.strip():
                raise VectorValidationError(
                    f"Record '{r.id}' has empty document text.",
                    collection=collection_name,
                    record_id=r.id,
                )
            if not r.embedding:
                raise VectorValidationError(
                    f"Record '{r.id}' has empty embedding vector.",
                    collection=collection_name,
                    record_id=r.id,
                )
            non_finite = [x for x in r.embedding if not math.isfinite(x)]
            if non_finite:
                raise VectorValidationError(
                    f"Record '{r.id}' contains non-finite values in embedding.",
                    collection=collection_name,
                    record_id=r.id,
                )
            for k, v in r.metadata.items():
                if not isinstance(v, (str, int, float, bool)):
                    raise VectorValidationError(
                        f"Record '{r.id}' metadata '{k}' is non-primitive: {type(v)}.",
                        collection=collection_name,
                        record_id=r.id,
                    )


def get_vector_store(
    settings: Optional[Settings] = None,
    client: Optional[Any] = None,
) -> VectorStore:
    """Instantiate and return the configured VectorStore implementation.

    Default: ChromaVectorStore.

    Args:
        settings: Application settings.
        client: Optional pre-configured client (e.g. for testing).

    Returns:
        VectorStore: Configured vector store instance.
    """
    app_settings = settings or get_settings()
    provider = getattr(app_settings, "VECTOR_STORE_PROVIDER", "chroma")
    if provider == "chroma":
        return ChromaVectorStore(settings=app_settings, client=client)

    raise VectorStoreError(
        f"Unsupported VECTOR_STORE_PROVIDER: '{provider}'. Supported: ['chroma']."
    )
