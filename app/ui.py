"""实现 V0.1 的文档上传、知识库提问和引用展示界面。"""

from __future__ import annotations

import os

import requests
import streamlit as st


DEFAULT_API_URL = os.environ.get("TRACERAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 120


def _upload_result_message(result: dict[str, object]) -> tuple[str, str]:
    """根据 API 的入库状态生成首次、重复或更新时的中文提示。"""

    file_name = str(result["file_name"])
    if result["already_indexed"]:
        return (
            "info",
            f"索引中已有该文件，未新增片段：{file_name}，"
            f"当前包含 {result['chunk_count']} 个片段。",
        )
    if result.get("replaced_existing", False):
        return (
            "success",
            f"已替换同名旧文档：{file_name}，解析 {result['page_count']} 页，"
            f"新版加入 {result['chunk_count']} 个片段。",
        )
    return (
        "success",
        f"索引完成：{file_name}，解析 {result['page_count']} 页，"
        f"新增 {result['chunk_count']} 个片段。",
    )


def main() -> None:
    """构建上传与查询界面，并将请求转发给 FastAPI。"""
    st.set_page_config(page_title="TraceRAG", page_icon="📚", layout="wide")
    st.title("TraceRAG")
    st.caption("上传资料，基于可追溯证据进行问答。")
    api_url = st.sidebar.text_input("FastAPI 地址", value=DEFAULT_API_URL).rstrip("/")

    st.subheader("1. 上传知识库文档")
    uploaded_file = st.file_uploader(
        "支持 PDF、Markdown 和 TXT（单文件最大 10 MiB）",
        type=["pdf", "md", "markdown", "txt"],
    )
    if st.button("上传并建立索引", disabled=uploaded_file is None):
        try:
            response = requests.post(
                f"{api_url}/documents",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.ok:
                result = response.json()
                message_type, message = _upload_result_message(result)
                if message_type == "info":
                    st.info(message)
                else:
                    st.success(message)
            else:
                st.error(f"上传失败（{response.status_code}）：{response.text}")
        except requests.RequestException as exc:
            st.error(f"无法连接 FastAPI：{exc}")

    st.divider()
    st.subheader("2. 提问")
    st.caption("系统会从已上传的文档中查找相关文字，依据找到的内容回答，并标明来源。")
    with st.form("query_form"):
        question = st.text_area("问题", placeholder="例如：年假政策是什么？")
        with st.expander("参考范围（Top-K，可选）"):
            st.write(
                "Top-K 表示每次提问最多交给模型参考、并在下方展示的片段数。"
                "默认 5 段，最终引用的段数可能更少。是否记载类问题会在内部多检索候选，"
                "但最终参考和展示仍不超过此上限。"
            )
            top_k = st.slider("最多供模型参考的文字段数", min_value=1, max_value=10, value=5)
            st.caption("调大可覆盖更多内容，也可能带入不相关的文字。")
        submitted = st.form_submit_button("获取回答")
    if submitted:
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            try:
                response = requests.post(
                    f"{api_url}/query",
                    json={"query": question, "top_k": top_k},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if not response.ok:
                    st.error(f"查询失败（{response.status_code}）：{response.text}")
                else:
                    result = response.json()
                    st.markdown("#### 回答")
                    st.write(result["answer"])
                    if result["rejected"]:
                        st.info("检索证据不足，系统已拒答。")
                    if result["citations"]:
                        st.markdown("#### 引用来源")
                        for index, citation in enumerate(result["citations"], start=1):
                            location = (
                                f"第 {citation['page_number']} 页"
                                if citation["page_number"] is not None
                                else "文本片段"
                            )
                            with st.expander(f"[{index}] {citation['file_name']} · {location}"):
                                st.write(citation["text"])
                    with st.expander("查看系统找到的相关文字"):
                        st.caption("这些是本次实际提供给模型的片段，数量不超过所选 Top-K；引用来源请看上方。")
                        for item in result["retrievals"]:
                            # 融合分用于排序，不能标为余弦相关度。
                            score_label = {"cosine": "余弦分数", "bm25": "BM25分数", "rrf": "RRF排序分"}.get(
                                item.get("score_kind", "cosine"), "排序分"
                            )
                            location = (
                                f"第 {item['page_number']} 页"
                                if item["page_number"] is not None
                                else "文本片段"
                            )
                            st.markdown(
                                f"**{item['file_name']} · {location} · "
                                f"候选排名 {item['candidate_rank'] or '—'} · "
                                f"{score_label} "
                                f"{item['score']:.3f}**"
                            )
                            st.write(item["text"])
            except requests.RequestException as exc:
                st.error(f"无法连接 FastAPI：{exc}")


if __name__ == "__main__":
    main()
