"""Small Streamlit UI for the V0.1 upload-and-query workflow."""

from __future__ import annotations

import os

import requests
import streamlit as st


DEFAULT_API_URL = os.environ.get("TRACERAG_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 120


def main() -> None:
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
                label = "已存在于索引" if result["already_indexed"] else "索引完成"
                st.success(
                    f"{label}：{result['file_name']}，解析 {result['page_count']} 页，"
                    f"加入 {result['chunk_count']} 个片段。"
                )
            else:
                st.error(f"上传失败（{response.status_code}）：{response.text}")
        except requests.RequestException as exc:
            st.error(f"无法连接 FastAPI：{exc}")

    st.divider()
    st.subheader("2. 提问")
    with st.form("query_form"):
        question = st.text_area("问题", placeholder="例如：年假政策是什么？")
        top_k = st.slider("检索证据数量 Top-K", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("查询")
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
                    with st.expander("调试：检索结果"):
                        for item in result["retrievals"]:
                            location = (
                                f"第 {item['page_number']} 页"
                                if item["page_number"] is not None
                                else "文本片段"
                            )
                            st.markdown(
                                f"**{item['file_name']} · {location} · "
                                f"相似度 {item['score']:.3f}**"
                            )
                            st.write(item["text"])
            except requests.RequestException as exc:
                st.error(f"无法连接 FastAPI：{exc}")


if __name__ == "__main__":
    main()
