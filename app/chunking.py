"""Deterministic text cleanup and fixed-window chunking with source metadata."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from app.models import Chunk, DocumentPage


_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")


def clean_text(text: str) -> str:
    """Normalize common extraction artifacts without changing meaningful text."""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTIPLE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[DocumentPage], *, chunk_size: int = 700, overlap: int = 100
) -> list[Chunk]:
    """Split pages into fixed character windows and assign stable traceable IDs."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[Chunk] = []
    stride = chunk_size - overlap
    for page in pages:
        content = clean_text(page.content)
        if not content:
            continue
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunk_index = len(chunks)
            identity = (
                f"{page.document_id}|{page.file_name}|{page.page_number}|"
                f"{chunk_index}|{chunk_text}"
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    content=chunk_text,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            if end == len(content):
                break
            start += stride

    return chunks
