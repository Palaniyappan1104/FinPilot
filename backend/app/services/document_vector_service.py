"""Document Vector Service for FinPilot (Phase 9.7: ChromaDB Storage).

Provides:
- chunk_embedding_to_vector_record: Convert ChunkEmbeddingResult to VectorRecord.
- store_document_embeddings: Store DocumentEmbeddingBatch into VectorStore.
- store_chunk_embeddings: Store ChunkEmbeddingResult list into VectorStore.

Preserves citation provenance using ChunkMetadata.to_vector_store_metadata()
and adheres to the deterministic collection strategy keyed by company/document.
"""

from typing import Dict, List, Optional, Union

from app.core.logging import get_logger
from app.models.documents import DocumentChunk
from app.models.embeddings import ChunkEmbeddingResult, DocumentEmbeddingBatch
from app.models.vector_store import VectorRecord, VectorStoreInsertionResult
from app.storage.vector_base import VectorStore, build_document_collection_name
from app.storage.vector_exceptions import VectorValidationError

logger = get_logger("app.services.document_vector_service")


def chunk_embedding_to_vector_record(
    result: ChunkEmbeddingResult,
    chunk_text: Optional[str] = None,
) -> VectorRecord:
    """Convert a ChunkEmbeddingResult into a VectorRecord for vector storage.

    Preserves full provenance metadata for downstream citations:
    - document_id, ticker, document_type, source_document
    - page_numbers, start_page, end_page
    - upload_date, chunk_index, section_name

    Args:
        result: ChunkEmbeddingResult containing embedding and provenance.
        chunk_text: Optional explicit chunk text if not stored on the result.

    Returns:
        VectorRecord: Strongly-typed vector record ready for vector store upsert.

    Raises:
        VectorValidationError: If text cannot be resolved or is empty.
    """
    text = chunk_text or getattr(result, "text", None)
    if not text or not text.strip():
        raise VectorValidationError(
            f"Cannot create VectorRecord for chunk '{result.chunk_id}': "
            "no document text provided.",
            record_id=result.chunk_id,
        )

    # Resolve flat primitive metadata
    if result.metadata is not None:
        metadata: Dict[str, Union[str, int, float, bool]] = (
            result.metadata.to_vector_store_metadata()
        )
    else:
        # Fallback reconstructing flat primitive citation metadata
        metadata = {
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "source_document": getattr(
                result, "source_document", f"{result.document_id}.pdf"
            ),
            "ticker": result.ticker,
            "document_type": result.document_type.value,
            "page_numbers": ",".join(str(p) for p in result.page_numbers),
            "start_page": result.start_page,
            "end_page": result.end_page,
            "upload_date": result.embedded_at.isoformat(),
            "chunk_index": result.chunk_index,
            "character_count": len(text),
            "word_count": len(text.split()),
        }
        if result.section_name is not None:
            metadata["section_name"] = result.section_name

    return VectorRecord(
        id=result.chunk_id,
        embedding=result.embedding,
        document=text,
        metadata=metadata,
    )


def store_document_embeddings(
    batch: DocumentEmbeddingBatch,
    vector_store: VectorStore,
    collection_name: Optional[str] = None,
    chunks: Optional[List[DocumentChunk]] = None,
) -> VectorStoreInsertionResult:
    """Store an entire DocumentEmbeddingBatch into the vector store.

    Uses a deterministic collection name keyed by company/document
    (finpilot_{ticker}_{document_id}) unless an explicit collection is provided.

    Args:
        batch: DocumentEmbeddingBatch containing embedding results for a document.
        vector_store: VectorStore implementation (e.g. ChromaVectorStore).
        collection_name: Optional override for collection name.
        chunks: Optional list of DocumentChunks for text resolution.

    Returns:
        VectorStoreInsertionResult: Insertion/upsert report.
    """
    col_name = collection_name or build_document_collection_name(
        ticker=batch.ticker,
        document_id=batch.document_id,
    )

    chunk_text_map: Dict[str, str] = {}
    if chunks:
        chunk_text_map = {c.chunk_id: c.text for c in chunks}

    records: List[VectorRecord] = []
    for res in batch.results:
        text = chunk_text_map.get(res.chunk_id) or res.text
        record = chunk_embedding_to_vector_record(res, chunk_text=text)
        records.append(record)

    logger.info(
        "Storing %d embedded chunks for document '%s' (%s) into collection '%s'.",
        len(records),
        batch.document_id,
        batch.ticker,
        col_name,
    )

    return vector_store.upsert_records(collection_name=col_name, records=records)


def store_chunk_embeddings(
    results: List[ChunkEmbeddingResult],
    vector_store: VectorStore,
    ticker: str,
    document_id: str,
    collection_name: Optional[str] = None,
    chunks: Optional[List[DocumentChunk]] = None,
) -> VectorStoreInsertionResult:
    """Store a list of ChunkEmbeddingResults into the vector store.

    Args:
        results: List of ChunkEmbeddingResult objects.
        vector_store: VectorStore instance.
        ticker: Stock ticker symbol.
        document_id: Source document identifier.
        collection_name: Optional override collection name.
        chunks: Optional list of DocumentChunks for text resolution.

    Returns:
        VectorStoreInsertionResult: Upsert summary.
    """
    col_name = collection_name or build_document_collection_name(
        ticker=ticker,
        document_id=document_id,
    )

    chunk_text_map: Dict[str, str] = {}
    if chunks:
        chunk_text_map = {c.chunk_id: c.text for c in chunks}

    records: List[VectorRecord] = []
    for res in results:
        text = chunk_text_map.get(res.chunk_id) or res.text
        record = chunk_embedding_to_vector_record(res, chunk_text=text)
        records.append(record)

    logger.info(
        "Storing %d chunk embeddings for '%s' into collection '%s'.",
        len(records),
        document_id,
        col_name,
    )

    return vector_store.upsert_records(collection_name=col_name, records=records)
