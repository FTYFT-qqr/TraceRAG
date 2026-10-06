"""组织中文资料库与证据问答工作台，复用 Streamlit 原生交互控件。"""

from __future__ import annotations

from html import escape
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote, urlsplit

import requests
import streamlit as st

from app.ui import _upload_result_message
from app.ui_styles import ARCHIVE_MARK, EMPTY_MARK, WORKSPACE_CSS

DEFAULT_API_URL = os.environ.get("TRACERAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 120
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _request(api_url: str, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """调用真实 API，统一错误提示并隐藏底层连接信息。"""
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.Timeout as exc:
        raise ValueError("服务响应超时，请稍后刷新；模型首次加载可能需要更长时间。") from exc
    except requests.RequestException as exc:
        raise ValueError("暂时无法连接服务，请检查服务地址和 API 是否已启动。") from exc
    if not response.ok:
        message = {
            404: "文档已不存在，或当前服务尚未提供资料管理接口，请刷新或升级 API。",
            409: "索引与当前模型配置不匹配，请恢复原模型配置或使用新的知识库目录。",
            413: "文件超过 10 MiB，请减小文件后再上传。",
            422: "输入未被服务接受，请检查文件格式、内容或问题长度。",
            502: "模型服务或索引处理失败，请稍后重试并检查服务配置。",
            503: "知识库服务暂未就绪，请检查模型配置、凭据及索引目录。",
        }.get(response.status_code, "操作未完成，请稍后重试。")
        raise ValueError(f"{message}（{response.status_code}）")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("服务返回了无法读取的结果，请检查 API 地址。") from exc
    if not isinstance(payload, dict):
        raise ValueError("服务返回的结果格式不正确。")
    return payload


def _catalog(api_url: str, *, force: bool = False) -> dict[str, Any] | None:
    """缓存当前会话的真实目录，失败不冒充空知识库。"""
    state = st.session_state
    if force or time.monotonic() - state.get("catalog_read_at", 0) > 15:
        try:
            current = _request(api_url, "GET", "/documents")
            if not isinstance(current.get("documents"), list):
                raise ValueError("服务返回的资料目录格式不正确。")
            if "catalog_generation" in state and state.catalog_generation != current.get("index_generation"):
                state.pop("answer_record", None)
            state.catalog = current
            state.catalog_generation = current.get("index_generation")
            state.catalog_error = None
        except ValueError as exc:
            state.catalog = None
            state.catalog_error = str(exc)
        state.catalog_read_at = time.monotonic()
    return state.get("catalog")


def _invalidate_answers() -> None:
    """资料变化后清除旧答案和候选，等待刷新目录。"""
    st.session_state.pop("answer_record", None)
    st.session_state.catalog_read_at = 0


def _headline(eyebrow: str, title: str, description: str) -> None:
    """展示统一标题层级及简短用途。"""
    st.markdown(f'<div class="tr-eyebrow">{escape(eyebrow)}</div><h1 class="tr-heading">{escape(title)}</h1>'
                f'<p class="tr-intro">{escape(description)}</p>', unsafe_allow_html=True)


def _empty(title: str, description: str) -> None:
    """展示可理解的空状态，不使用虚构数据。"""
    st.markdown(f'<div class="tr-empty"><div class="tr-empty-symbol">{EMPTY_MARK}</div>'
                f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def _go_library() -> None:
    """在组件回调阶段导航，避免修改已实例化的单选控件状态。"""
    st.session_state.workspace_page = "资料库"


def _sidebar() -> tuple[str, str, bool]:
    """提供导航及服务设置，切换服务时清除此前结果。"""
    with st.sidebar:
        st.markdown(f'<div class="tr-brand"><div class="tr-mark">{ARCHIVE_MARK}</div><div>'
                    '<div class="tr-brand-name">TraceRAG</div><div class="tr-brand-caption">让每个答案都有出处</div></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="tr-section-label">工作空间</div>', unsafe_allow_html=True)
        page = st.radio("工作空间", ["知识问答", "资料库"], label_visibility="collapsed", key="workspace_page")
        st.divider()
        with st.expander("连接与设置"):
            api_url = st.text_input("服务地址", value=DEFAULT_API_URL, key="api_address").strip().rstrip("/")
            st.caption("填写 FastAPI 地址；容器 UI 通常使用 http://api:8000。")
        try:
            parsed = urlsplit(api_url)
            valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                     and not parsed.username and not parsed.password
                     and parsed.port != 0 and not parsed.query and not parsed.fragment)
        except ValueError:
            valid = False
        if st.session_state.get("active_api_url") != api_url:
            for key in ("catalog", "catalog_error", "catalog_generation", "catalog_read_at", "answer_record", "health_read_at"):
                st.session_state.pop(key, None)
            st.session_state.active_api_url = api_url
        if valid and time.monotonic() - st.session_state.get("health_read_at", 0) > 15:
            was_healthy = st.session_state.get("service_healthy", False)
            try:
                response = requests.get(f"{api_url}/health", timeout=4)
                healthy = response.ok and response.json().get("status") == "ok"
            except (requests.RequestException, ValueError, AttributeError):
                healthy = False
            if was_healthy and not healthy:
                # 服务掉线时旧回答可能已不对应现存资料；重连后必须重读目录。
                st.session_state.pop("answer_record", None)
                st.session_state.catalog_read_at = 0
            st.session_state.service_healthy = healthy
            st.session_state.health_read_at = time.monotonic()
        connected = valid and st.session_state.get("service_healthy", False)
        label = "API 已连接" if connected else "服务未连接"
        dot_class = "" if connected else "offline"
        st.markdown(f'<div class="tr-status"><span class="tr-dot {dot_class}"></span>{label}</div>', unsafe_allow_html=True)
        if not valid:
            st.warning("请输入不含凭据的 http 或 https 服务地址。")
        if st.button("重新连接", icon=":material/refresh:", use_container_width=True):
            st.session_state.health_read_at = 0
            st.session_state.catalog_read_at = 0
            st.rerun()
        st.caption("答案由知识库证据支持。找不到足够依据时，系统会明确告知。")
    return page, api_url, connected


def _upload(api_url: str, enabled: bool) -> None:
    """上传文档，反馈入库状态并刷新目录和来源。"""
    with st.container(border=True, key="upload_panel"):
        st.subheader("添加资料")
        st.caption("支持 PDF、Markdown、TXT，单文件最大 10 MiB。")
        file = st.file_uploader("选择资料文件", type=["pdf", "md", "markdown", "txt"], key="document_upload", max_upload_size=10, disabled=not enabled)
        if file and file.size > MAX_UPLOAD_BYTES:
            st.warning("文件超过 10 MiB，请减小文件后再上传。")
        if st.button("上传并入库", type="primary", icon=":material/upload:", use_container_width=True,
                     disabled=not enabled or file is None or file.size > MAX_UPLOAD_BYTES):
            try:
                with st.spinner("正在解析资料并建立索引…"):
                    result = _request(api_url, "POST", "/documents", files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")})
                kind, message = _upload_result_message(result)
                if not result.get("already_indexed"):
                    _invalidate_answers()
                st.session_state.flash = (kind, message)
                st.session_state.catalog_read_at = 0
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "上传结果格式不正确，请刷新目录确认结果。")
        st.caption("同名文件会更新内容；完全相同的资料会自动跳过。")


@st.dialog("移除知识库资料")
def _confirm_delete(api_url: str, document: dict[str, Any]) -> None:
    """确认删除当前文档，操作失败时保留说明。"""
    st.write("即将从当前知识库移除以下资料：")
    st.text(document["file_name"])
    st.caption(f"包含 {document['chunk_count']} 个片段。移除后不再用于新问题；需要时可重新上传。历史快照仍按恢复策略保留。")
    cancel, confirm = st.columns(2)
    if cancel.button("保留资料", use_container_width=True):
        st.rerun()
    if confirm.button("确认移除", type="primary", use_container_width=True, icon=":material/delete:"):
        try:
            with st.spinner("正在更新知识库…"):
                result = _request(api_url, "DELETE", f"/documents/{quote(str(document['document_id']), safe='')}")
            _invalidate_answers()
            st.session_state.flash = ("success", f"已移除 {result['file_name']}，剩余 {result['document_count']} 份资料。")
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc) if isinstance(exc, ValueError) else "删除结果格式不正确，请刷新目录确认结果。")


def _documents(api_url: str, catalog: dict[str, Any] | None, connected: bool) -> None:
    """展示真实资料目录、名称搜索、格式筛选及删除确认。"""
    _headline("LIBRARY / 资料管理", "把知识库整理好", "添加、查找和管理资料，让每次问答使用当前有效的内容。")
    if catalog:
        st.markdown(f'<div class="tr-summary"><div><strong>{catalog["document_count"]}</strong><span>已入库资料</span></div>'
                    f'<div><strong>{catalog["chunk_count"]}</strong><span>可检索片段</span></div></div>', unsafe_allow_html=True)
    listing, upload = st.columns([2.1, 1], gap="large")
    with upload:
        _upload(api_url, connected)
    with listing:
        st.subheader("知识库资料")
        filter_name, filter_type, refresh = st.columns([2, 1, .8], vertical_alignment="bottom")
        term = filter_name.text_input("搜索文件名", placeholder="输入资料名称…", key="document_search")
        kind = filter_type.selectbox("文件格式", ["全部格式", "PDF", "Markdown", "TXT"])
        if refresh.button("刷新", icon=":material/refresh:", use_container_width=True):
            st.session_state.catalog_read_at = 0
            st.session_state.health_read_at = 0
            st.rerun()
        if catalog is None:
            st.warning(st.session_state.get("catalog_error") or "连接服务后即可查看资料库。")
            return
        documents = catalog["documents"]
        suffixes = {"PDF": {".pdf"}, "Markdown": {".md", ".markdown"}, "TXT": {".txt"}}
        visible = [document for document in documents if term.casefold() in document["file_name"].casefold()
                   and (kind == "全部格式" or Path(document["file_name"]).suffix.lower() in suffixes[kind])]
        if not documents:
            with st.container(border=True, key="empty_library"):
                _empty("还没有入库资料", "添加第一份资料后，你就可以提问并核对答案的原文来源。")
        elif not visible:
            _empty("没有找到匹配资料", "试试其他名称，或将格式筛选改为全部格式。")
        else:
            page_count = (len(visible) + 11) // 12
            page = st.selectbox("资料页码", range(1, page_count + 1), format_func=lambda value: f"第 {value} / {page_count} 页") if page_count > 1 else 1
            st.caption(f"共 {len(visible)} 份匹配资料")
            for document in visible[(page - 1) * 12:page * 12]:
                with st.container(border=True, key=f"document_{document['document_id']}"):
                    name, action = st.columns([5, 1.2], vertical_alignment="center")
                    suffix = Path(document["file_name"]).suffix.lstrip(".").upper()
                    pages = document.get("indexed_page_count")
                    meta = f"{document['chunk_count']} 个片段" + (f" · {pages} 页已索引" if pages else " · 文本资料")
                    name.markdown(f'<div class="tr-file"><span class="tr-file-type">{escape(suffix)}</span>'
                                  f'<div><div class="tr-file-name">{escape(document["file_name"])}</div>'
                                  f'<div class="tr-file-meta">{escape(meta)}</div></div></div>', unsafe_allow_html=True)
                    if action.button("移除", icon=":material/delete:", key=f"remove_{document['document_id']}", use_container_width=True, disabled=not connected):
                        _confirm_delete(api_url, document)
            st.caption("应用保存切分文本及向量，不另存上传原文件。移除针对当前知识库，历史快照仍保留。")


def _answer(record: dict[str, Any]) -> None:
    """并排呈现回答与对应来源，折叠检索技术详情。"""
    result = record["result"]
    answer_column, sources = st.columns([1.65, 1], gap="large")
    with answer_column:
        with st.container(border=True, key="answer_panel"):
            refused = bool(result["rejected"])
            label = "目前缺少足够依据" if refused else "已找到资料依据"
            status_class = "refused" if refused else ""
            st.markdown(f'<div class="tr-answer-label {status_class}">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tr-question">{escape(record["question"])}</div>', unsafe_allow_html=True)
            st.markdown(result["answer"])
            st.caption("可以补充相关资料，或尝试更具体的问法。" if refused else "答案保留来源原文表述，编号对应右侧引用依据。")
            if st.button("清除本次回答", icon=":material/close:"):
                st.session_state.pop("answer_record", None)
                st.rerun()
    with sources:
        st.subheader(f"引用依据 · {len(result['citations'])}")
        if not result["citations"]:
            st.caption("本次未返回来源引用。")
        for index, citation in enumerate(result["citations"], 1):
            page = f"第 {citation['page_number']} 页" if citation.get("page_number") else "文本片段"
            with st.container(border=True, key=f"source_{index}"):
                st.markdown(f'<span class="tr-source-number">来源 {index:02d} / [{index}]</span>'
                            f'<div class="tr-source-name">{escape(citation["file_name"])}</div>'
                            f'<div class="tr-source-meta">{page}</div>', unsafe_allow_html=True)
                with st.expander(f"查看原文 [{index}]"):
                    st.write(citation["text"])
                    st.caption(f"片段标识：{citation['chunk_id']}")
    with st.expander("检索详情 · 排名与相关文字"):
        st.caption("以下是实际供模型参考的片段。排序分用于检查检索过程，不能当作答案正确率。")
        for index, item in enumerate(result["retrievals"], 1):
            label = {"cosine": "余弦分", "bm25": "BM25 分", "rrf": "RRF 排序分"}.get(item.get("score_kind"), "排序分")
            st.write(f"{index}. {item['file_name']} · 原候选排名 {item.get('candidate_rank') or '—'} · {label} {item['score']:.4f}")
            raw = [f"{name} {item[key]:.4f}" for key, name in (("vector_score", "原始余弦"), ("bm25_score", "原始 BM25")) if item.get(key) is not None]
            if raw:
                st.caption(" · ".join(raw))
            st.write(item["text"])


def _questions(api_url: str, catalog: dict[str, Any] | None, connected: bool) -> None:
    """突出提问与阅读，保存最近回答并解释参考片段上限。"""
    _headline("WORKSPACE / 知识问答", "从资料中，找到有出处的答案", "提出问题，阅读答案，再沿着引用回到原文。")
    if catalog:
        st.caption(f"当前知识库 · {catalog['document_count']} 份资料 · {catalog['chunk_count']} 个可检索片段")
    if connected and catalog is None:
        st.warning(st.session_state.get("catalog_error") or "目录暂不可用；上传与问答仍可尝试。")
    empty = catalog is not None and not catalog["documents"]
    with st.container(border=True, key="question_panel"):
        with st.form("query_form", border=False):
            question = st.text_area("你想了解什么？", placeholder="例如：设备预约需要哪些手续？", max_chars=8000)
            with st.expander("参考片段上限 · 默认 5 段"):
                top_k = st.slider("最多参考多少段文字", min_value=1, max_value=10, value=5)
                st.caption("每次最多供模型参考并展示的片段数量（Top-K）。默认 5 段；实际引用可能更少。调大可能覆盖更多内容，也可能带入无关文字。")
            submitted = st.form_submit_button("查找答案", type="primary", icon=":material/arrow_forward:", disabled=not connected or empty)
        if empty:
            st.caption("知识库还没有资料，请先到资料库添加文档。")
            st.button("前往资料库", icon=":material/folder_open:", on_click=_go_library)
    if submitted:
        if not question.strip():
            st.warning("先写下一个问题，再查找答案。")
        else:
            try:
                with st.spinner("正在检索资料并核对回答依据…"):
                    result = _request(api_url, "POST", "/query", json={"query": question.strip(), "top_k": top_k})
                if not all(key in result for key in ("answer", "rejected", "citations", "retrievals")):
                    raise ValueError("回答结果格式不完整，请重试。")
                st.session_state.answer_record = {"question": question.strip(), "result": result}
            except ValueError as exc:
                st.session_state.pop("answer_record", None)
                st.error(str(exc))
    if record := st.session_state.get("answer_record"):
        _answer(record)
    else:
        _empty("答案和依据，会一起呈现", "这里会显示你的问题、资料中的回答，以及可以展开核对的原文引用。")
    st.markdown('<div class="tr-footer">TraceRAG · 依据原文回答 · 来源可追溯 · 证据不足时拒答</div>', unsafe_allow_html=True)


def run_workspace() -> None:
    """组织服务连接、真实目录及问答工作区。"""
    st.set_page_config(page_title="TraceRAG · 知识工作台", page_icon=":material/library_books:", layout="wide")
    st.markdown(WORKSPACE_CSS, unsafe_allow_html=True)
    page, api_url, connected = _sidebar()
    catalog = None
    if connected:
        with st.spinner("正在读取知识库…"):
            catalog = _catalog(api_url)
    if flash := st.session_state.pop("flash", None):
        (st.info if flash[0] == "info" else st.success)(flash[1])
    if page == "资料库":
        _documents(api_url, catalog, connected)
    else:
        _questions(api_url, catalog, connected)
