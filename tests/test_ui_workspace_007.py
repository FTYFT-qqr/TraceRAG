"""以隔离 API 驱动真实 Streamlit 控件，验收 007 的前端操作。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote

import pytest
from streamlit.testing.v1 import AppTest

from app import ui_workspace


@dataclass
class _FakeApi:
    """在内存中模拟文档目录和问答，避免触碰用户知识库。"""

    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    contents: dict[str, bytes] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    generation: int = 0
    fail_catalog: bool = False

    def add_document(self, name: str, content: bytes) -> None:
        """准备隔离资料并更新目录代数。"""

        document_id = name.replace(".", "_")
        self.documents[document_id] = {
            "document_id": document_id,
            "file_name": name,
            "chunk_count": 2,
            "indexed_page_count": 1 if name.lower().endswith(".pdf") else None,
        }
        self.contents[document_id] = content
        self.generation += 1

    def request(self, _api_url: str, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """返回与前端契约一致的目录、上传、删除及问答结果。"""

        self.calls.append((method, path))
        if method == "GET" and path == "/documents":
            if self.fail_catalog:
                raise ValueError("暂时无法读取资料目录。")
            documents = list(self.documents.values())
            return {
                "documents": documents,
                "document_count": len(documents),
                "chunk_count": sum(item["chunk_count"] for item in documents),
                "index_generation": str(self.generation),
            }
        if method == "POST" and path == "/documents":
            name, content, _mime_type = kwargs["files"]["file"]
            document_id = name.replace(".", "_")
            previous = self.contents.get(document_id)
            if previous != content:
                self.add_document(name, content)
            return {
                "file_name": name,
                "already_indexed": previous == content,
                "replaced_existing": previous is not None and previous != content,
                "page_count": 1,
                "chunk_count": 2,
            }
        if method == "DELETE" and path.startswith("/documents/"):
            document_id = unquote(path.rsplit("/", 1)[-1])
            document = self.documents.pop(document_id)
            self.contents.pop(document_id)
            self.generation += 1
            return {"file_name": document["file_name"], "document_count": len(self.documents)}
        if method == "POST" and path == "/query":
            return {
                "answer": "当前资料有依据。[1]",
                "rejected": False,
                "citations": [{
                    "file_name": next(iter(self.documents.values()))["file_name"],
                    "page_number": 1,
                    "text": "当前资料有依据。",
                    "chunk_id": "sample:1",
                }],
                "retrievals": [],
            }
        raise AssertionError(f"不应调用的请求：{method} {path}")


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, _FakeApi, dict[str, bool]]:
    """装配可交互的真实页面和可控制的健康状态。"""

    fake_api = _FakeApi()
    health = {"online": True}
    monkeypatch.setattr(ui_workspace, "_request", fake_api.request)
    monkeypatch.setattr(
        ui_workspace.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=health["online"], json=lambda: {"status": "ok" if health["online"] else "offline"}
        ),
    )
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui.py"), default_timeout=10)
    return app, fake_api, health


def _choose_library(app: AppTest) -> None:
    """通过导航控件进入资料库并执行一次页面重跑。"""

    app.radio[0].set_value("资料库").run()
    assert not app.exception


def _labelled(widgets: Any, label: str) -> Any:
    """按可见标签查找组件，避免依赖 Streamlit 自动生成的内部键。"""

    return next(widget for widget in widgets if widget.label == label)


def _ask(app: AppTest) -> None:
    """提交一个问题，建立等待失效的旧回答。"""

    # 表单字段和提交按钮需在同一次 AppTest 运行中发送，兼容 Streamlit 1.64/1.65。
    app.text_area[0].set_value("当前资料是什么？")
    _labelled(app.button, "查找答案").click().run()
    assert not app.exception
    assert app.session_state["answer_record"]["result"]["answer"] == "当前资料有依据。[1]"


def test_upload_duplicate_and_same_name_replacement_clear_answers(workspace: tuple[AppTest, _FakeApi, dict[str, bool]]) -> None:
    """上传、重复跳过和同名替换均从真实控件触发，并检查旧答案。"""

    app, fake_api, _health = workspace
    fake_api.add_document("原有.txt", b"old")
    app.run()
    _ask(app)
    _choose_library(app)

    app.file_uploader[0].set_value(("指南.txt", b"first", "text/plain")).run()
    _labelled(app.button, "上传并入库").click().run()
    assert ("POST", "/documents") in fake_api.calls
    assert "answer_record" not in app.session_state
    assert any("索引完成：指南.txt" in item.value for item in app.success)

    app.session_state["answer_record"] = {"question": "尚有效的旧问题"}
    app.file_uploader[0].set_value(("指南.txt", b"first", "text/plain")).run()
    _labelled(app.button, "上传并入库").click().run()
    assert any("未新增片段" in item.value for item in app.info)
    assert fake_api.generation == 2
    assert "answer_record" in app.session_state

    app.file_uploader[0].set_value(("指南.txt", b"second", "text/plain")).run()
    _labelled(app.button, "上传并入库").click().run()
    assert any("已替换同名旧文档" in item.value for item in app.success)
    assert fake_api.generation == 3
    assert app.session_state["catalog"]["document_count"] == 2
    assert "answer_record" not in app.session_state


def test_name_format_filter_and_cancel_confirm_delete(workspace: tuple[AppTest, _FakeApi, dict[str, bool]]) -> None:
    """验证名称及格式组合筛选、取消移除和最终确认。"""

    app, fake_api, _health = workspace
    fake_api.add_document("设备规则.pdf", b"pdf")
    fake_api.add_document("设备维护.md", b"md")
    fake_api.add_document("采购说明.txt", b"txt")
    app.run()
    _choose_library(app)

    _labelled(app.text_input, "搜索文件名").set_value("设备").run()
    _labelled(app.selectbox, "文件格式").set_value("PDF").run()
    page = "\n".join(item.value for item in app.markdown)
    assert "设备规则.pdf" in page
    assert "设备维护.md" not in page
    assert "采购说明.txt" not in page
    assert any("共 1 份匹配资料" in item.value for item in app.caption)

    _labelled(app.button, "移除").click().run()
    assert not app.exception
    assert len(fake_api.documents) == 3
    assert any(button.label == "确认移除" for button in app.button)
    _labelled(app.button, "保留资料").click().run()
    assert len(fake_api.documents) == 3

    # AppTest 不执行弹窗 fragment 的二次点击；直接驱动同一弹窗函数的按钮。
    dialog_script = (
        "from app import ui_workspace\n"
        "ui_workspace._confirm_delete.__wrapped__("
        "'http://127.0.0.1:8000', " + repr(fake_api.documents["设备规则_pdf"]) + ")"
    )
    cancel_dialog = AppTest.from_string(dialog_script, default_timeout=10).run()
    _labelled(cancel_dialog.button, "保留资料").click().run()
    assert not cancel_dialog.exception
    assert not any(method == "DELETE" for method, _path in fake_api.calls)
    assert len(fake_api.documents) == 3

    dialog = AppTest.from_string(dialog_script, default_timeout=10).run()
    assert not dialog.exception
    dialog.session_state["answer_record"] = {"question": "旧问题"}
    _labelled(dialog.button, "确认移除").click().run()
    assert not dialog.exception
    assert "设备规则_pdf" not in fake_api.documents
    assert len(fake_api.documents) == 2
    assert "answer_record" not in dialog.session_state
    assert dialog.session_state["flash"][1].startswith("已移除 设备规则.pdf")


def test_generation_change_and_connection_failure_do_not_show_stale_answer(workspace: tuple[AppTest, _FakeApi, dict[str, bool]]) -> None:
    """外部资料变化和失联后，页面不得继续展示旧问答。"""

    app, fake_api, health = workspace
    fake_api.add_document("规范.txt", b"v1")
    app.run()
    _ask(app)
    fake_api.add_document("新增.txt", b"v2")
    app.session_state["catalog_read_at"] = 0
    app.run()
    assert "answer_record" not in app.session_state

    _ask(app)
    health["online"] = False
    app.session_state["health_read_at"] = 0
    app.run()
    assert any("服务未连接" in item.value for item in app.markdown)
    assert _labelled(app.button, "查找答案").disabled
    assert "answer_record" not in app.session_state
    _choose_library(app)
    assert app.file_uploader[0].disabled
    assert _labelled(app.button, "上传并入库").disabled
    assert all(button.disabled for button in app.button if button.label == "移除")


def test_catalog_error_is_not_rendered_as_empty_library(workspace: tuple[AppTest, _FakeApi, dict[str, bool]]) -> None:
    """目录读取失败时提示连接问题，避免误报为没有资料。"""

    app, fake_api, _health = workspace
    fake_api.add_document("规范.txt", b"content")
    app.run()
    _choose_library(app)
    fake_api.fail_catalog = True
    app.session_state["catalog_read_at"] = 0
    app.run()
    assert any("暂时无法读取资料目录" in item.value for item in app.warning)
    assert not any("还没有入库资料" in item.value for item in app.markdown)


def test_workspace_css_contains_narrow_screen_layout() -> None:
    """窄屏断点保证资料卡、问答卡和移除按钮可纵向操作。"""

    css = ui_workspace.WORKSPACE_CSS
    assert "@media(max-width:900px)" in css
    assert "min-width:100%!important" in css
    assert "min-height:44px" in css
