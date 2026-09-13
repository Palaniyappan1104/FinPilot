"""Document Embedding Service for FinPilot (Phase 9.6).

Provides:
- chunk_to_embedding_request: Convert a DocumentChunk to a ChunkEmbeddingRequest.
- embed_chunked_document: Generate embeddings for all chunks in a ChunkedDocument.
- DocumentEmbedder: Service class wrapping an EmbeddingProvider for document batches.

This service sits between the Phase 9.4/9.5 chunking pipeline and the Phase 9.7
ChromaDB storage layer. It is fully independent of ChromaDB.
"""

from typing import List

from app.core.logging import get_logger
from app.models.documents import ChunkedDocument, DocumentChunk
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
    DocumentEmbeddingBatch,
    EmbeddingModelInfo,
)
from app.providers.embedding import EmbeddingProvider

logger = get_logger("app.services.document_embedder")


# ===========================================================================
# HELPER: DocumentChunk -> ChunkEmbeddingRequest
# ===========================================================================


def chunk_to_embedding_request(chunk: DocumentChunk) -> ChunkEmbeddingRequest:
    """Convert a DocumentChunk from Phase 9.4/9.5 into a ChunkEmbeddingRequest.

    Preserves full provenance (chunk_id, document_id, ticker, document_type,
    page numbers, section, metadata) alongside the text to be embedded.

    Args:
        chunk: DocumentChunk instance from Phase 9.4 or 9.5.

    Returns:
        ChunkEmbeddingRequest: Validated input model for the embedding provider.
    """
    return ChunkEmbeddingRequest(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        ticker=chunk.ticker,
        document_type=chunk.document_type,
        text=chunk.text,
        page_numbers=list(chunk.page_numbers),
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        chunk_index=chunk.chunk_index,
        section_name=chunk.section_name,
        metadata=chunk.metadata,
    )


# ===========================================================================
# CONVENIENCE FUNCTION: Embed a ChunkedDocument
# ===========================================================================


def embed_chunked_document(
    chunked_doc: ChunkedDocument,
    provider: EmbeddingProvider,
) -> DocumentEmbeddingBatch:
    """Generate embeddings for every chunk in a ChunkedDocument.

    Preserves the same document provenance fields (document_id, ticker,
    document_type) in the returned DocumentEmbeddingBatch so that the
    Phase 9.7 storage layer has everything it needs without reading back
    from the source models.

    Args:
        chunked_doc: ChunkedDocument from Phase 9.4 or 9.5.
        provider: Configured EmbeddingProvider instance.

    Returns:
        DocumentEmbeddingBatch: Batch of ChunkEmbeddingResult objects.
    """
    requests: List[ChunkEmbeddingRequest] = [
        chunk_to_embedding_request(c) for c in chunked_doc.chunks
    ]

    logger.info(
        "Embedding %d chunks for document '%s' (%s) via %s/%s",
        len(requests),
        chunked_doc.document_id,
        chunked_doc.ticker,
        provider.provider_name,
        provider.model_name,
    )

    results: List[ChunkEmbeddingResult] = provider.embed_batch(requests)

    return DocumentEmbeddingBatch(
        document_id=chunked_doc.document_id,
        ticker=chunked_doc.ticker,
        document_type=chunked_doc.document_type,
        total_chunks=len(results),
        embedding_model=provider.model_name,
        embedding_dimensions=provider.dimensions,
        results=results,
    )


# ===========================================================================
# SERVICE CLASS: DocumentEmbedder
# ===========================================================================


class DocumentEmbedder:
    """Service class wrapping an EmbeddingProvider for document embedding tasks.

    Provides both single-chunk and full-document embedding, keeps provider
    isolated, and logs operational details without exposing secrets.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Initialize with a concrete EmbeddingProvider instance.

        Args:
            provider: Any EmbeddingProvider implementation (Gemini, mock, etc.).
        """
        self._provider = provider
        logger.debug(
            "DocumentEmbedder initialized with provider=%s model=%s dims=%d",
            provider.provider_name,
            provider.model_name,
            provider.dimensions,
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        """Return the model info for the backing provider."""
        return self._provider.get_model_info()

    def embed_chunk(self, chunk: DocumentChunk) -> ChunkEmbeddingResult:
        """Generate an embedding for a single DocumentChunk.

        Args:
            chunk: DocumentChunk instance.

        Returns:
            ChunkEmbeddingResult: Provenance-coupled embedding result.
        """
        request = chunk_to_embedding_request(chunk)
        return self._provider.embed_single(request)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[ChunkEmbeddingResult]:
        """Generate embeddings for a list of DocumentChunks.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            List[ChunkEmbeddingResult]: Ordered embedding results.
        """
        if not chunks:
            return []
        requests = [chunk_to_embedding_request(c) for c in chunks]
        return self._provider.embed_batch(requests)

    def embed_document(self, chunked_doc: ChunkedDocument) -> DocumentEmbeddingBatch:
        """Generate embeddings for every chunk in a ChunkedDocument.

        Args:
            chunked_doc: ChunkedDocument from Phase 9.4/9.5.

        Returns:
            DocumentEmbeddingBatch: Complete batch of embedding results.
        """
        return embed_chunked_document(chunked_doc, self._provider)
