"""读取 PDF、Markdown 和 TXT，并统一保留文件名及 PDF 页码。"""

from __future__ import annotations

import hashlib
from io import BytesIO

from pypdf import PdfReader

from app.models import DocumentPage, UnsupportedDocumentError


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def _safe_file_name(file_name: str) -> str:
    """去掉客户端传入路径，只保留文件名，避免写入任意目录。"""

    normalized = file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("A valid file name is required.")
    return normalized


def _decode_text(data: bytes, file_name: str) -> str:
    """优先使用 UTF-8，兼容常见的 Windows GB18030 文本编码。"""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            # 中文 Windows 文本常使用 GB18030 保存，UTF-8 解码失败后再尝试此编码。
            return data.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{file_name} is not valid UTF-8 or GB18030 text.") from exc


def load_document(file_name: str, data: bytes) -> list[DocumentPage]:
    """解析受支持的单个文件，并为内容和来源生成稳定身份。"""

    safe_name = _safe_file_name(file_name)
    suffix = "." + safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or '(none)'}'. "
            "Supported types: PDF, Markdown, TXT."
        )
    if not data:
        raise ValueError(f"{safe_name} is empty.")

    # 同名文件只要字节内容变化，document_id 就会变化，用于识别版本更新。
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
                # 页码从 1 开始，保持与 PDF 阅读器和用户习惯一致。
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
