"""保留界面启动入口及上传结果提示，页面组件位于 ui_workspace。"""

from __future__ import annotations


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
    """启动资料管理与证据问答工作台，保持现有 Streamlit 入口。"""
    from app.ui_workspace import run_workspace

    run_workspace()


if __name__ == "__main__":
    main()
