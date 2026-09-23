from __future__ import annotations

from app.document_loader import load_document
from app.models import UnsupportedDocumentError


def _make_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny valid PDF with text content without another test dependency."""

    font_id = 3 + len(page_texts) * 2
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{3 + index * 2} 0 R".encode() for index in range(len(page_texts)))
        + f"] /Count {len(page_texts)} >>".encode(),
    ]
    for index, page_text in enumerate(page_texts):
        page_id = 3 + index * 2
        content_id = page_id + 1
        text_bytes = page_text.encode("ascii")
        stream = b"BT /F1 12 Tf 72 720 Td (" + text_bytes + b") Tj ET"
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode(),
                b"<< /Length "
                + str(len(stream)).encode()
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            ]
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def test_load_pdf_extracts_page_text_and_source_metadata() -> None:
    pages = load_document("handbook.pdf", _make_pdf(["Leave policy", "Holiday policy"]))

    assert [page.content.strip() for page in pages] == ["Leave policy", "Holiday policy"]
    assert [page.page_number for page in pages] == [1, 2]
    assert {page.file_name for page in pages} == {"handbook.pdf"}
    assert len({page.document_id for page in pages}) == 1


def test_load_markdown_and_text_files_as_source_aware_documents() -> None:
    markdown = load_document("folder\\policy.MD", b"# Leave\n10 days")
    plain_text = load_document("policy.txt", "年假：10天".encode("gb18030"))

    assert markdown[0].file_name == "policy.MD"
    assert markdown[0].page_number is None
    assert markdown[0].content == "# Leave\n10 days"
    assert plain_text[0].content == "年假：10天"


def test_document_id_changes_when_file_content_changes() -> None:
    first = load_document("policy.txt", b"Version one")[0]
    second = load_document("policy.txt", b"Version two")[0]

    assert first.document_id != second.document_id


def test_loader_rejects_unsupported_empty_and_malformed_files() -> None:
    import pytest

    with pytest.raises(UnsupportedDocumentError):
        load_document("policy.docx", b"not supported")
    with pytest.raises(ValueError, match="empty"):
        load_document("empty.txt", b"")
    with pytest.raises(ValueError, match="Could not read PDF"):
        load_document("broken.pdf", b"not a pdf")
