"""Vector store domain models (Phase 9.7: ChromaDB Storage).

Defines:
- VectorRecord: Vector record with ID, vector, document text, and metadata.
- VectorStoreInsertionResult: Result of an insert/upsert operation.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Union

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
