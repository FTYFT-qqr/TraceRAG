"""验证上传成功、完全重复和同名更新时的界面提示文案。"""

from app.ui import _upload_result_message


def test_upload_ui_reports_new_chunks_for_first_upload() -> None:
    """验证界面准确显示首次上传新建的片段数量。"""
    message_type, message = _upload_result_message(
        {
            "file_name": "guide.pdf",
            "already_indexed": False,
            "replaced_existing": False,
            "page_count": 3,
            "chunk_count": 5,
        }
    )

    assert message_type == "success"
    assert "新增 5 个片段" in message


def test_upload_ui_reports_no_new_chunks_for_duplicate_upload() -> None:
    """验证界面说明重复上传没有新增片段。"""
    message_type, message = _upload_result_message(
        {"file_name": "guide.pdf", "already_indexed": True, "chunk_count": 5}
    )

    assert message_type == "info"
    assert "未新增片段" in message
    assert "当前包含 5 个片段" in message


def test_upload_ui_reports_replacement_for_same_name_update() -> None:
    """验证同名文档更新时界面显示替换结果。"""
    message_type, message = _upload_result_message(
        {
            "file_name": "guide.pdf",
            "already_indexed": False,
            "replaced_existing": True,
            "page_count": 4,
            "chunk_count": 6,
        }
    )

    assert message_type == "success"
    assert "已替换同名旧文档" in message
    assert "新版加入 6 个片段" in message
