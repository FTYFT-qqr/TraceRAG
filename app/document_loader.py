"""Load PDF, Markdown, and text files into source-aware page records."""

from __future__ import annotations

import hashlib
from io import BytesIO

from pypdf import PdfReader

from app.models import DocumentPage, UnsupportedDocumentError


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def _safe_file_name(file_name: str) -> str:
    """Keep only the uploaded file's name, regardless of client path syntax."""

    normalized = file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("A valid file name is required.")
    return normalized


def _decode_text(data: bytes, file_name: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            # Chinese Windows documents are often saved as GB18030.
            return data.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{file_name} is not valid UTF-8 or GB18030 text.") from exc


def load_document(file_name: str, data: bytes) -> list[DocumentPage]:
    """Parse one supported document while retaining file and PDF page metadata."""

    safe_name = _safe_file_name(file_name)
    suffix = "." + safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or '(none)'}'. "
            "Supported types: PDF, Markdown, TXT."
        )
    if not data:
        raise ValueError(f"{safe_name} is empty.")

    document_id = hashlib.sha256(safe_name.encode("utf-8") + b"\0" + data).hexdigest()
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise ValueError(f"{safe_name} is encrypted and cannot be loaded.")
            return [
                DocumentPage(
                    document_id=document_id,
                    file_name=safe_name,
                    page_number=page_index,
                    content=page.extract_text() or "",
                )
                for page_index, page in enumerate(reader.pages, start=1)
            ]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read PDF document {safe_name}.") from exc

    text = _decode_text(data, safe_name)
    return [
        DocumentPage(
            document_id=document_id,
            file_name=safe_name,
            page_number=None,
            content=text,
        )
    ]
