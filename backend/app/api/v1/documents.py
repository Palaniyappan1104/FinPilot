"""Document upload API endpoints for FinPilot Research Vault (Phase 9.1)."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.documents import DocumentType, DocumentUploadResponse
from app.services.document_validator import (
    DocumentCorruptedFormatError,
    DocumentEmptyError,
    DocumentFileSizeExceededError,
    DocumentUnsupportedTypeError,
    DocumentValidationError,
    InvalidTickerError,
    validate_document_file,
    validate_ticker,
)
from app.storage.base import DocumentStorage
from app.storage.exceptions import StorageIOError, StoragePathTraversalError
from app.storage.local import LocalDocumentStorage

logger = get_logger("app.api.v1.documents")

router = APIRouter()


def get_document_storage() -> DocumentStorage:
    """Dependency provider for document storage backend."""
    settings = get_settings()
    return LocalDocumentStorage(base_directory=settings.DOCUMENTS_STORAGE_PATH)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload research document",
    description=(
        "Upload a financial research document (e.g. 10-K, 10-Q, earnings transcript, "
        "investor presentation) associated with a stock ticker."
    ),
    operation_id="uploadResearchDocument",
)
@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload"),
    ticker: str = Form(..., description="Stock ticker symbol (e.g. AAPL, NVDA)"),
    document_type: DocumentType = Form(
        ..., description="Category of the financial document"
    ),
    storage: DocumentStorage = Depends(get_document_storage),
) -> DocumentUploadResponse:
    """Handle multipart file upload, validate metadata/content, and store locally."""
    # 1. Validate ticker
    try:
        clean_ticker = validate_ticker(ticker)
    except InvalidTickerError as e:
        logger.warning("Upload rejected: invalid ticker '%s' (%s)", ticker, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # 2. Read file content safely
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {e}",
        ) from e

    # 3. Validate file metadata and contents
    try:
        validate_document_file(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )
    except DocumentFileSizeExceededError as e:
        logger.warning(
            "Upload rejected: file '%s' exceeds max size (%d > %d bytes)",
            file.filename,
            e.file_size,
            e.max_size,
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(e),
        ) from e
    except (
        DocumentEmptyError,
        DocumentUnsupportedTypeError,
        DocumentCorruptedFormatError,
        DocumentValidationError,
    ) as e:
        logger.warning("Upload rejected for '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # 4. Save to storage backend
    try:
        stored_doc = storage.save(
            content=content,
            original_filename=file.filename or "document.pdf",
            ticker=clean_ticker,
            document_type=document_type,
            mime_type=file.content_type or "application/pdf",
        )
    except StoragePathTraversalError as e:
        logger.error("Path traversal security violation on upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal attempted.",
        ) from e
    except StorageIOError as e:
        logger.error("Storage I/O failure storing '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage error: unable to save document.",
        ) from e

    logger.info(
        "Successfully uploaded document id='%s', ticker='%s', type='%s', bytes=%d",
        stored_doc.document_id,
        stored_doc.ticker,
        stored_doc.document_type.value,
        stored_doc.file_size_bytes,
    )

    return DocumentUploadResponse(
        document_id=stored_doc.document_id,
        ticker=stored_doc.ticker,
        document_type=stored_doc.document_type,
        original_filename=stored_doc.original_filename,
        storage_path=stored_doc.storage_path,
        file_size_bytes=stored_doc.file_size_bytes,
        upload_status="stored",
        uploaded_at=stored_doc.uploaded_at,
    )
