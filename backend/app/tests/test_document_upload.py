"""Phase 9.1 Document Upload unit and API integration tests.

Verifies:
- 9.1.1 Upload API contract (file, ticker association, DocumentType enum).
- 9.1.2 Document storage abstraction (DocumentStorage, LocalDocumentStorage).
- 9.1.3 Validation rules (extension, size limits, PDF sanity checks).
- Security safeguards (path traversal, filename sanitization, UUID generation).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import status
from starlette.testclient import TestClient

from app.api.v1.documents import get_document_storage
from app.core.config import Settings
from app.main import create_application
from app.models.documents import DocumentType, DocumentUploadResponse, StoredDocument
from app.services.document_validator import (
    DocumentCorruptedFormatError,
    DocumentEmptyError,
    DocumentFileSizeExceededError,
    DocumentUnsupportedTypeError,
    InvalidTickerError,
    validate_document_file,
    validate_ticker,
)
from app.storage.exceptions import (
    StorageIOError,
    StorageNotFoundError,
    StoragePathTraversalError,
)
from app.storage.local import LocalDocumentStorage

VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog /Pages 2 0 R >>\n"
    b"endobj\n"
    b"2 0 obj\n"
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
    b"endobj\n"
    b"3 0 obj\n"
    b"<< /Type /Page /Parent 2 0 R >>\n"
    b"endobj\n"
    b"xref\n"
    b"0 4\n"
    b"trailer\n"
    b"<< /Root 1 0 R >>\n"
    b"%%EOF\n"
)


# ===========================================================================
# 1. STORAGE UNIT TESTS (9.1.2)
# ===========================================================================


class TestLocalDocumentStorage:
    """Test suite for LocalDocumentStorage filesystem implementation."""

    def test_storage_init_creates_directory(self, tmp_path: Path):
        """Storage initialization creates the target base directory safely."""
        target_dir = tmp_path / "deep" / "nested" / "storage"
        assert not target_dir.exists()

        storage = LocalDocumentStorage(base_directory=target_dir)
        assert target_dir.is_dir()
        assert storage.base_directory == target_dir.resolve()

    def test_storage_save_success(self, tmp_path: Path):
        """Saving document writes bytes, generates unique ID, and returns metadata."""
        storage = LocalDocumentStorage(base_directory=tmp_path)

        stored = storage.save(
            content=VALID_PDF_BYTES,
            original_filename="apple_10k_2023.pdf",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            mime_type="application/pdf",
        )

        assert isinstance(stored, StoredDocument)
        assert stored.ticker == "AAPL"
        assert stored.document_type == DocumentType.ANNUAL_REPORT
        assert stored.original_filename == "apple_10k_2023.pdf"
        assert stored.file_size_bytes == len(VALID_PDF_BYTES)
        assert stored.document_id.startswith("doc_")
        assert len(stored.sha256_checksum) == 64

        # Verify physical file existence
        disk_path = storage.get_filesystem_path(stored.storage_path)
        assert disk_path is not None
        assert disk_path.is_file()
        assert disk_path.read_bytes() == VALID_PDF_BYTES

    def test_storage_collision_resistant_ids(self, tmp_path: Path):
        """Saving duplicate filenames generates collision-resistant IDs."""
        storage = LocalDocumentStorage(base_directory=tmp_path)

        doc1 = storage.save(
            content=VALID_PDF_BYTES,
            original_filename="report.pdf",
            ticker="MSFT",
            document_type=DocumentType.COMPANY_REPORT,
        )
        doc2 = storage.save(
            content=VALID_PDF_BYTES,
            original_filename="report.pdf",
            ticker="MSFT",
            document_type=DocumentType.COMPANY_REPORT,
        )

        assert doc1.document_id != doc2.document_id
        assert doc1.storage_path != doc2.storage_path
        assert storage.exists(doc1.storage_path)
        assert storage.exists(doc2.storage_path)

    def test_storage_get_bytes_success(self, tmp_path: Path):
        """Stored bytes can be retrieved by storage path."""
        storage = LocalDocumentStorage(base_directory=tmp_path)
        doc = storage.save(
            content=b"%PDF-sample",
            original_filename="test.pdf",
            ticker="NVDA",
            document_type=DocumentType.INVESTOR_PRESENTATION,
        )

        retrieved = storage.get_bytes(doc.storage_path)
        assert retrieved == b"%PDF-sample"

    def test_storage_get_bytes_not_found(self, tmp_path: Path):
        """Reading non-existent storage path raises StorageNotFoundError."""
        storage = LocalDocumentStorage(base_directory=tmp_path)
        with pytest.raises(StorageNotFoundError, match="Document not found"):
            storage.get_bytes("NONEXISTENT/doc_missing.pdf")

    def test_storage_exists_and_delete(self, tmp_path: Path):
        """Document existence and deletion behavior."""
        storage = LocalDocumentStorage(base_directory=tmp_path)
        doc = storage.save(
            content=VALID_PDF_BYTES,
            original_filename="delete_me.pdf",
            ticker="GOOGL",
            document_type=DocumentType.OTHER_SUPPORTED,
        )

        assert storage.exists(doc.storage_path) is True
        deleted = storage.delete(doc.storage_path)
        assert deleted is True
        assert storage.exists(doc.storage_path) is False

        # Deleting non-existent returns False
        assert storage.delete(doc.storage_path) is False

    def test_storage_path_traversal_prevention(self, tmp_path: Path):
        """Path traversal patterns in storage path raise StoragePathTraversalError."""
        storage = LocalDocumentStorage(base_directory=tmp_path)

        with pytest.raises(StoragePathTraversalError, match="Path traversal detected"):
            storage.get_bytes("../../../secret.txt")

        with pytest.raises(StoragePathTraversalError, match="Path traversal detected"):
            storage.get_bytes("AAPL/../../etc/passwd")

    def test_storage_filename_sanitization_removes_path_traversal(self, tmp_path: Path):
        """Client-provided filenames with directory traversal are sanitized."""
        storage = LocalDocumentStorage(base_directory=tmp_path)

        dangerous_name = "../../etc/passwd/../../../malicious.pdf"
        doc = storage.save(
            content=VALID_PDF_BYTES,
            original_filename=dangerous_name,
            ticker="AMZN",
            document_type=DocumentType.EARNINGS_TRANSCRIPT,
        )

        # Original filename is stripped of path separators
        assert ".." not in doc.original_filename
        assert "/" not in doc.original_filename
        assert "\\" not in doc.original_filename
        assert doc.original_filename.endswith("malicious.pdf")

        # Saved file is strictly inside tmp_path / AMZN
        disk_path = storage.get_filesystem_path(doc.storage_path)
        assert disk_path is not None
        assert disk_path.is_file()
        assert disk_path.parent == tmp_path / "AMZN"

    def test_storage_null_byte_rejected(self, tmp_path: Path):
        """Null bytes in storage path raise StoragePathTraversalError."""
        storage = LocalDocumentStorage(base_directory=tmp_path)
        with pytest.raises(StoragePathTraversalError, match="null bytes"):
            storage.get_bytes("AAPL/doc_\x00evil.pdf")

    def test_storage_io_error_handling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Underlying OSError triggers StorageIOError."""
        storage = LocalDocumentStorage(base_directory=tmp_path)

        def mock_write_bytes(self, data):
            raise OSError("Disk full simulation")

        monkeypatch.setattr(Path, "write_bytes", mock_write_bytes)

        with pytest.raises(StorageIOError, match="Failed to write document"):
            storage.save(
                content=VALID_PDF_BYTES,
                original_filename="test.pdf",
                ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
            )


# ===========================================================================
# 2. VALIDATION UNIT TESTS (9.1.3)
# ===========================================================================


class TestDocumentValidation:
    """Test suite for document upload validation rules."""

    def test_validate_ticker_valid(self):
        """Valid tickers are accepted and normalized to uppercase."""
        assert validate_ticker("aapl") == "AAPL"
        assert validate_ticker("NVDA") == "NVDA"
        assert validate_ticker("  msft  ") == "MSFT"
        assert validate_ticker("brk.a") == "BRK.A"
        assert validate_ticker("bf-b") == "BF-B"

    def test_validate_ticker_invalid(self):
        """Empty, whitespace, or malformed tickers raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError, match="empty or whitespace"):
            validate_ticker("")

        with pytest.raises(InvalidTickerError, match="empty or whitespace"):
            validate_ticker("   ")

        with pytest.raises(InvalidTickerError, match="Invalid ticker symbol"):
            validate_ticker("AAPL;DROP TABLE")

        with pytest.raises(InvalidTickerError, match="Invalid ticker symbol"):
            validate_ticker("VERYLONGTICKERNAME")

    def test_validate_document_valid_pdf(self):
        """Valid PDF passes all validation checks."""
        validate_document_file(
            filename="quarterly_report.pdf",
            content=VALID_PDF_BYTES,
            content_type="application/pdf",
        )

    def test_validate_document_empty_file(self):
        """0-byte file raises DocumentEmptyError."""
        with pytest.raises(DocumentEmptyError, match="empty \\(0 bytes\\)"):
            validate_document_file(
                filename="empty.pdf",
                content=b"",
                content_type="application/pdf",
            )

    def test_validate_document_file_size_exceeded(self):
        """File larger than max_size_bytes raises DocumentFileSizeExceededError."""
        content = VALID_PDF_BYTES + (b"X" * 1000)
        with pytest.raises(DocumentFileSizeExceededError, match="exceeds maximum"):
            validate_document_file(
                filename="large.pdf",
                content=content,
                content_type="application/pdf",
                max_size_bytes=50,
            )

    def test_validate_document_unsupported_extension(self):
        """Non-PDF file extensions raise DocumentUnsupportedTypeError."""
        with pytest.raises(DocumentUnsupportedTypeError, match="is unsupported"):
            validate_document_file(
                filename="notes.txt",
                content=b"some notes",
                content_type="text/plain",
            )

        with pytest.raises(DocumentUnsupportedTypeError, match="is unsupported"):
            validate_document_file(
                filename="script.exe",
                content=b"binary",
            )

    def test_validate_document_missing_extension(self):
        """Filename without extension raises DocumentUnsupportedTypeError."""
        with pytest.raises(DocumentUnsupportedTypeError, match="is unsupported"):
            validate_document_file(
                filename="document_without_extension",
                content=VALID_PDF_BYTES,
            )

    def test_validate_document_unsupported_mime(self):
        """Explicit non-PDF MIME type raises DocumentUnsupportedTypeError."""
        with pytest.raises(DocumentUnsupportedTypeError, match="Content-Type"):
            validate_document_file(
                filename="fake.pdf",
                content=VALID_PDF_BYTES,
                content_type="image/jpeg",
            )

    def test_validate_document_corrupted_pdf_missing_magic_bytes(self):
        """File missing '%PDF-' header raises DocumentCorruptedFormatError."""
        fake_content = b"This is just a text file renamed to fake.pdf"
        with pytest.raises(
            DocumentCorruptedFormatError, match="missing '%PDF-' header"
        ):
            validate_document_file(
                filename="fake.pdf",
                content=fake_content,
                content_type="application/pdf",
            )

    def test_validate_document_too_small_for_pdf(self):
        """File smaller than 8 bytes raises DocumentCorruptedFormatError."""
        with pytest.raises(DocumentCorruptedFormatError, match="too small"):
            validate_document_file(
                filename="tiny.pdf",
                content=b"%PDF",
                content_type="application/pdf",
            )


# ===========================================================================
# 3. API INTEGRATION TESTS (9.1.1)
# ===========================================================================


class TestDocumentUploadAPI:
    """Integration test suite for FastAPI /api/v1/documents/upload endpoint."""

    @pytest.fixture
    def test_client(self, tmp_path: Path):
        """Provide TestClient with temporary document storage directory."""
        test_storage = LocalDocumentStorage(base_directory=tmp_path)
        app = create_application()
        app.dependency_overrides[get_document_storage] = lambda: test_storage
        with TestClient(app) as client:
            yield client
        app.dependency_overrides.clear()

    def test_api_valid_pdf_upload_success(self, test_client: TestClient):
        """Uploading valid PDF returns 201 Created and DocumentUploadResponse schema."""
        files = {
            "file": ("nvda_q2_transcript.pdf", VALID_PDF_BYTES, "application/pdf"),
        }
        data = {
            "ticker": "NVDA",
            "document_type": "earnings_transcript",
        }

        response = test_client.post("/api/v1/documents/upload", files=files, data=data)

        assert response.status_code == status.HTTP_201_CREATED
        body = response.json()
        assert body["ticker"] == "NVDA"
        assert body["document_type"] == "earnings_transcript"
        assert body["original_filename"] == "nvda_q2_transcript.pdf"
        assert body["document_id"].startswith("doc_")
        assert body["upload_status"] == "stored"
        assert body["file_size_bytes"] == len(VALID_PDF_BYTES)
        assert "storage_path" in body
        assert "uploaded_at" in body

        # Validate with Pydantic model
        validated_resp = DocumentUploadResponse.model_validate(body)
        assert validated_resp.document_type == DocumentType.EARNINGS_TRANSCRIPT

    def test_api_upload_alias_route(self, test_client: TestClient):
        """Both /api/v1/documents/upload and /api/v1/documents respond identically."""
        files = {
            "file": ("presentation.pdf", VALID_PDF_BYTES, "application/pdf"),
        }
        data = {
            "ticker": "AAPL",
            "document_type": "investor_presentation",
        }

        response = test_client.post("/api/v1/documents", files=files, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["ticker"] == "AAPL"

    @pytest.mark.parametrize(
        "doc_type",
        [
            "annual_report",
            "investor_presentation",
            "earnings_transcript",
            "company_report",
            "other_supported",
        ],
    )
    def test_api_all_document_types_supported(
        self, test_client: TestClient, doc_type: str
    ):
        """All controlled DocumentType enum variants are accepted."""
        files = {
            "file": ("doc.pdf", VALID_PDF_BYTES, "application/pdf"),
        }
        data = {
            "ticker": "MSFT",
            "document_type": doc_type,
        }

        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["document_type"] == doc_type

    def test_api_missing_file(self, test_client: TestClient):
        """Missing file payload returns 422 Unprocessable Entity."""
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_api_missing_ticker(self, test_client: TestClient):
        """Missing ticker returns 422 Unprocessable Entity."""
        files = {"file": ("report.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_api_missing_document_type(self, test_client: TestClient):
        """Missing document_type returns 422 Unprocessable Entity."""
        files = {"file": ("report.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "AAPL"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_api_invalid_document_type_enum(self, test_client: TestClient):
        """Unsupported document_type value returns 422 Unprocessable Entity."""
        files = {"file": ("report.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "AAPL", "document_type": "unsupported_type_xyz"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_api_invalid_ticker_format(self, test_client: TestClient):
        """Invalid ticker symbol returns 400 Bad Request."""
        files = {"file": ("report.pdf", VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "INVALID!@#$", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid ticker symbol" in response.json()["error"]["message"]

    def test_api_empty_file_upload(self, test_client: TestClient):
        """Uploading empty 0-byte file returns 400 Bad Request."""
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty (0 bytes)" in response.json()["error"]["message"]

    def test_api_file_size_exceeded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Uploading file exceeding MAX_UPLOAD_SIZE_BYTES returns 413."""
        test_storage = LocalDocumentStorage(base_directory=tmp_path)
        # Mock settings with small max upload size of 20 bytes
        mock_settings = Settings(MAX_UPLOAD_SIZE_BYTES=20)
        monkeypatch.setattr(
            "app.services.document_validator.get_settings",
            lambda: mock_settings,
        )

        app = create_application()
        app.dependency_overrides[get_document_storage] = lambda: test_storage

        with TestClient(app) as client:
            files = {"file": ("oversized.pdf", VALID_PDF_BYTES, "application/pdf")}
            data = {"ticker": "AAPL", "document_type": "annual_report"}
            response = client.post("/api/v1/documents/upload", files=files, data=data)

            assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
            err_msg = response.json()["error"]["message"]
            assert "exceeds maximum allowed limit" in err_msg

    def test_api_unsupported_file_extension(self, test_client: TestClient):
        """Uploading non-PDF file extension returns 400 Bad Request."""
        files = {"file": ("notes.docx", b"PK...", "application/vnd.openxmlformats")}
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "is unsupported" in response.json()["error"]["message"]

    def test_api_corrupted_pdf_magic_bytes(self, test_client: TestClient):
        """Uploading corrupted/disguised PDF returns 400 Bad Request."""
        files = {
            "file": (
                "corrupted.pdf",
                b"NOT_A_PDF_DOCUMENT_CONTENT",
                "application/pdf",
            )
        }
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "missing '%PDF-' header" in response.json()["error"]["message"]

    def test_api_path_traversal_filename_sanitized(self, test_client: TestClient):
        """Path traversal in filename is sanitized and upload succeeds cleanly."""
        dangerous_filename = "../../../../evil_payload.pdf"
        files = {"file": (dangerous_filename, VALID_PDF_BYTES, "application/pdf")}
        data = {"ticker": "AAPL", "document_type": "annual_report"}
        response = test_client.post("/api/v1/documents/upload", files=files, data=data)

        assert response.status_code == status.HTTP_201_CREATED
        body = response.json()
        assert ".." not in body["original_filename"]
        assert body["original_filename"] == "evil_payload.pdf"

    def test_api_storage_io_error_handling(self, tmp_path: Path):
        """Storage I/O failure gracefully returns 500 Internal Server Error."""
        mock_storage = MagicMock()
        mock_storage.save.side_effect = StorageIOError("Disk drive failure")

        app = create_application()
        app.dependency_overrides[get_document_storage] = lambda: mock_storage

        with TestClient(app) as client:
            files = {"file": ("valid.pdf", VALID_PDF_BYTES, "application/pdf")}
            data = {"ticker": "AAPL", "document_type": "annual_report"}
            response = client.post("/api/v1/documents/upload", files=files, data=data)

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "unable to save document" in response.json()["error"]["message"]
