"""统一配置 API 和运行时共用的结构化日志格式。"""

from __future__ import annotations

import logging


def configure_logging(level_name: str) -> None:
    """设置时间、级别和模块名，便于从服务日志定位问题。"""

    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        # 替换开发环境中可能已存在的默认 handler，确保格式保持一致。
        force=True,
    )
