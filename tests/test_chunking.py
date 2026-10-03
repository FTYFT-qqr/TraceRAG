"""验证抽取文本清洗、分块重叠和来源元数据保留。"""

import pytest

from app.chunking import chunk_pages, clean_text
from app.document_loader import load_document
from app.models import DocumentPage


def test_clean_text_normalizes_extraction_whitespace() -> None:
    """验证文本清理可以统一抽取过程产生的空格和换行。"""
    assert clean_text("  A\u00a0  B\r\n\r\n\r\nC\u200b ") == "A B\n\nC"


def test_chunk_pages_preserves_metadata_and_overlap() -> None:
    """用短文本明确检查相邻窗口共享的字符范围。"""
    page = DocumentPage(
        document_id="doc-1",
        file_name="handbook.pdf",
        page_number=4,
        content="0123456789ABCDEFGHIJ",
    )

    chunks = chunk_pages([page], chunk_size=10, overlap=3)

    assert [chunk.content for chunk in chunks] == ["0123456789", "789ABCDEFG", "EFGHIJ"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.file_name == "handbook.pdf" for chunk in chunks)
    assert all(chunk.page_number == 4 for chunk in chunks)
    assert chunks[0].content[-3:] == chunks[1].content[:3]
    assert chunks[1].content[-3:] == chunks[2].content[:3]


def test_chunk_ids_are_stable_and_document_content_is_traceable() -> None:
    """验证相同输入产生稳定 Chunk ID 且保留来源信息。"""
    pages = load_document("policy.txt", b"Annual leave policy " * 10)

    first = chunk_pages(pages, chunk_size=32, overlap=8)
    second = chunk_pages(pages, chunk_size=32, overlap=8)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.file_name == "policy.txt" for chunk in first)
    assert all(chunk.document_id == pages[0].document_id for chunk in first)


def test_chunk_pages_skips_empty_text_and_validates_window_sizes() -> None:
    """验证切分跳过空页并拒绝无效窗口参数。"""
    empty_page = DocumentPage("doc", "empty.txt", None, "  \n\t ")
    assert chunk_pages([empty_page]) == []

    with pytest.raises(ValueError, match="positive"):
        chunk_pages([empty_page], chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="smaller"):
        chunk_pages([empty_page], chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_pages([empty_page], chunk_size=10, overlap=-1)


def test_chunks_receive_contiguous_indexes_across_pages() -> None:
    """确认 chunk_index 跨 PDF 页连续递增，页码仍各自保留。"""
    pages = [
        DocumentPage("doc", "policy.pdf", 1, "a" * 12),
        DocumentPage("doc", "policy.pdf", 2, "b" * 12),
    ]

    chunks = chunk_pages(pages, chunk_size=8, overlap=2)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]
