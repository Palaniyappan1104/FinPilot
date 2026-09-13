"""Typed models for document chunk embeddings (Phase 9.6).

Defines:
- EmbeddingModelInfo: Model and dimension metadata.
- ChunkEmbeddingRequest: Input model for embedding a DocumentChunk.
- ChunkEmbeddingResult: Output model coupling chunk provenance with its vector.
- DocumentEmbeddingBatch: Container for a batch of ChunkEmbeddingResult objects.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.documents import ChunkMetadata, DocumentType


class EmbeddingModelInfo(BaseModel):
    """Metadata about the configured embedding model and vector dimensions."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(description="Embedding provider identifier (e.g. 'gemini')")
    model_name: str = Field(description="Embedding model identifier string")
    dimensions: int = Field(ge=1, description="Expected output vector dimensionality")


class ChunkEmbeddingRequest(BaseModel):
    """Input request for embedding a single document chunk.

    Carries the text to be embedded together with full provenance so the
    result can be assembled without losing the source document context.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique chunk identifier from DocumentChunk")
    document_id: str = Field(description="Source document identifier")
    ticker: str = Field(description="Uppercase stock ticker / company identifier")
    document_type: DocumentType = Field(description="Document category enum value")
    text: str = Field(description="Raw text content to embed")
    page_numbers: List[int] = Field(
        description="1-indexed source physical page numbers spanning this chunk"
    )
    start_page: int = Field(ge=1, description="First source page of this chunk")
    end_page: int = Field(ge=1, description="Final source page of this chunk")
    chunk_index: int = Field(
        ge=1, description="1-indexed sequence position of chunk in document"
    )
    section_name: Optional[str] = Field(
        default=None,
        description="Detected section heading if identified",
    )
    metadata: Optional[ChunkMetadata] = Field(
        default=None,
        description="Full ChunkMetadata from Phase 9.5 if available",
    )

    @field_validator("text")
    @classmethod
    def text_must_be_non_empty(cls, v: str) -> str:
        """Reject blank or whitespace-only text input."""
        if not v or not v.strip():
            raise ValueError(
                "ChunkEmbeddingRequest.text must be a non-empty, non-whitespace string."
            )
        return v


class ChunkEmbeddingResult(BaseModel):
    """Output model coupling a DocumentChunk's provenance with its embedding vector.

    Designed to be ChromaDB-ready: carries every field needed by the Phase 9.7
    storage layer without coupling to ChromaDB itself.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Source document identifier")
    ticker: str = Field(description="Uppercase stock ticker / company identifier")
    document_type: DocumentType = Field(description="Document category enum value")
    page_numbers: List[int] = Field(
        description="1-indexed source physical page numbers spanning this chunk"
    )
    start_page: int = Field(ge=1, description="First source page of this chunk")
    end_page: int = Field(ge=1, description="Final source page of this chunk")
    chunk_index: int = Field(
        ge=1, description="1-indexed sequence position of chunk in document"
    )
    section_name: Optional[str] = Field(
        default=None,
        description="Detected section heading if identified",
    )
    metadata: Optional[ChunkMetadata] = Field(
        default=None,
        description="Full ChunkMetadata from Phase 9.5 if available",
    )
    embedding: List[float] = Field(
        description="Dense numeric embedding vector for this chunk"
    )
    embedding_model: str = Field(
        description="Model identifier used to produce this embedding"
    )
    embedding_dimensions: int = Field(
        ge=1, description="Dimensionality of the embedding vector"
    )
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the embedding was generated",
    )

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_valid(cls, v: List[float]) -> List[float]:
        """Validate the embedding vector is non-empty and all values are finite."""
        import math

        if not v:
            raise ValueError("Embedding vector must be non-empty.")
        non_finite = [x for x in v if not math.isfinite(x)]
        if non_finite:
            raise ValueError(
                f"Embedding vector contains {len(non_finite)} non-finite value(s) "
                f"(NaN or infinity). All values must be finite floats."
            )
        return v

    @model_validator(mode="after")
    def validate_dimension_consistency(self) -> "ChunkEmbeddingResult":
        """Ensure embedding_dimensions matches the actual vector length."""
        if self.embedding_dimensions != len(self.embedding):
            raise ValueError(
                f"embedding_dimensions ({self.embedding_dimensions}) must match "
                f"actual vector length ({len(self.embedding)})."
            )
        return self


class DocumentEmbeddingBatch(BaseModel):
    """Container for the embedding results of an entire document's chunks."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Source document identifier")
    ticker: str = Field(description="Uppercase stock ticker symbol")
    document_type: DocumentType = Field(description="Category of source document")
    total_chunks: int = Field(ge=0, description="Total number of embedded chunks")
    embedding_model: str = Field(
        description="Embedding model identifier used for all chunks"
    )
    embedding_dimensions: int = Field(
        ge=1, description="Dimensionality of all embedding vectors in this batch"
    )
    results: List[ChunkEmbeddingResult] = Field(
        description="Ordered list of ChunkEmbeddingResult objects"
    )
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the batch was completed",
    )

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> "DocumentEmbeddingBatch":
        """Ensure total_chunks and per-result dimensions match batch metadata."""
        if self.total_chunks != len(self.results):
            raise ValueError(
                f"total_chunks ({self.total_chunks}) must match "
                f"number of results ({len(self.results)})."
            )
        for i, res in enumerate(self.results):
            if res.embedding_dimensions != self.embedding_dimensions:
                raise ValueError(
                    f"Result at index {i} has dimension "
                    f"{res.embedding_dimensions}, expected "
                    f"{self.embedding_dimensions}."
                )
        return self
