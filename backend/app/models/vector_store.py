"""Vector store domain models (Phase 9.7: Storage, Phase 9.9: Similarity Search).

Defines:
- VectorRecord: Vector record with ID, vector, document text, and metadata.
- VectorStoreInsertionResult: Result of an insert/upsert operation.
- VectorSearchResult: Match item from similarity search with distance & provenance.
- SimilaritySearchResults: Container for search results with metadata.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VectorRecord(BaseModel):
    """Vector-store-agnostic record coupling a chunk ID, vector, text, and metadata.

    Carries primitive metadata conforming to vector store constraints (e.g. ChromaDB
    requires flat primitive values: str, int, float, bool).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique record/chunk identifier")
    embedding: List[float] = Field(description="Dense numeric embedding vector")
    document: str = Field(
        description="Raw chunk text content associated with this vector"
    )
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Flat primitive metadata dictionary",
    )

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only record IDs."""
        if not v or not v.strip():
            raise ValueError("VectorRecord.id must be a non-empty string.")
        return v.strip()

    @field_validator("document")
    @classmethod
    def document_must_be_non_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only document text."""
        if not v or not v.strip():
            raise ValueError("VectorRecord.document text must be non-empty.")
        return v

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_valid(cls, v: List[float]) -> List[float]:
        """Validate embedding vector is non-empty and all elements are finite floats."""
        if not v:
            raise ValueError("VectorRecord.embedding must be non-empty.")
        non_finite = [x for x in v if not math.isfinite(x)]
        if non_finite:
            raise ValueError(
                f"Embedding contains {len(non_finite)} non-finite value(s)."
            )
        return v

    @field_validator("metadata")
    @classmethod
    def metadata_values_must_be_primitives(
        cls, v: Dict[str, Union[str, int, float, bool]]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Validate all metadata values are str, int, float, or bool."""
        for key, val in v.items():
            if not isinstance(val, (str, int, float, bool)):
                raise ValueError(
                    f"Metadata key '{key}' has non-primitive value of type "
                    f"{type(val).__name__}. Must be str, int, float, or bool."
                )
        return v


class VectorStoreInsertionResult(BaseModel):
    """Result of an insert or upsert operation in the vector store."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Target vector collection name")
    record_ids: List[str] = Field(description="List of record IDs inserted or updated")
    total_records: int = Field(
        ge=0, description="Total number of records affected in this operation"
    )
    inserted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the operation",
    )


class VectorSearchResult(BaseModel):
    """Structured result representing a single match from a similarity search.

    Preserves the stored document chunk, full citation metadata, and the raw
    distance value returned by the underlying vector store.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique chunk/record identifier")
    distance: float = Field(
        description=(
            "Distance metric returned by vector store (e.g. cosine distance; "
            "lower values indicate higher semantic similarity)"
        )
    )
    document: str = Field(description="Raw document chunk text stored in vector store")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Flat primitive metadata stored with the record",
    )
    distance_metric: str = Field(
        default="cosine_distance",
        description=(
            "Explicit identifier of the distance metric "
            "(e.g. 'cosine_distance', 'l2_distance')"
        ),
    )

    @field_validator("id")
    @classmethod
    def id_must_be_non_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only result IDs."""
        if not v or not v.strip():
            raise ValueError("VectorSearchResult.id must be a non-empty string.")
        return v.strip()

    @field_validator("distance")
    @classmethod
    def distance_must_be_finite(cls, v: float) -> float:
        """Validate that the distance score is a finite float."""
        if not math.isfinite(v):
            raise ValueError("VectorSearchResult.distance must be a finite float.")
        return v

    @property
    def document_id(self) -> Optional[str]:
        """Source document ID from metadata if present."""
        val = self.metadata.get("document_id")
        return str(val) if val is not None else None

    @property
    def ticker(self) -> Optional[str]:
        """Stock ticker symbol from metadata if present."""
        val = self.metadata.get("ticker")
        return str(val) if val is not None else None

    @property
    def document_type(self) -> Optional[str]:
        """Document category string from metadata if present."""
        val = self.metadata.get("document_type")
        return str(val) if val is not None else None

    @property
    def page_numbers(self) -> List[int]:
        """Parsed source page numbers from metadata if present."""
        raw = self.metadata.get("page_numbers")
        if not raw:
            start = self.metadata.get("start_page")
            end = self.metadata.get("end_page")
            if isinstance(start, int) and isinstance(end, int):
                return list(range(start, end + 1))
            return []
        if isinstance(raw, str):
            try:
                return [int(p.strip()) for p in raw.split(",") if p.strip()]
            except ValueError:
                return []
        if isinstance(raw, int):
            return [raw]
        return []

    @property
    def start_page(self) -> Optional[int]:
        """First source page from metadata if present."""
        val = self.metadata.get("start_page")
        return int(val) if isinstance(val, (int, float)) else None

    @property
    def end_page(self) -> Optional[int]:
        """Last source page from metadata if present."""
        val = self.metadata.get("end_page")
        return int(val) if isinstance(val, (int, float)) else None

    @property
    def section_name(self) -> Optional[str]:
        """Section heading from metadata if present."""
        val = self.metadata.get("section_name")
        return str(val) if val is not None else None


class SimilaritySearchResults(BaseModel):
    """Container for the results of a similarity search query (Phase 9.9).

    Couples query context with the ordered list of VectorSearchResult matches.
    """

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Target vector store collection name")
    results: List[VectorSearchResult] = Field(
        default_factory=list,
        description="Ordered list of search results matching the query vector",
    )
    total_results: int = Field(
        ge=0,
        description="Total number of results returned",
    )
    distance_metric: str = Field(
        default="cosine_distance",
        description="Distance metric reported by vector store",
    )
    searched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when search was executed",
    )
