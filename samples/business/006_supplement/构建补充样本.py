"""为 006 整改后回归生成全新的气象观测站模拟业务样本。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = "岚峰气象观测站运维规则.txt"
QUESTIONS = "questions_supplement.json"

# 每项对应一个不同的信息需求；同句设置真实的相近数值干扰。
ANSWERABLE = [
    (
        "岚峰主站的海拔是多少米？",
        "岚峰主站海拔2860米，备用站海拔3120米。",
        [["主站海拔2860米", "2860米"]],
        ["numeric", "near_value", "direct"],
    ),
    (
        "冬季外箱加热器在什么气温开启、什么气温关闭？",
        "冬季外箱加热器在气温降至负18摄氏度时开启，升至负12摄氏度时关闭。",
        [["负18摄氏度", "-18摄氏度", "零下18摄氏度"],
         ["负12摄氏度", "-12摄氏度", "零下12摄氏度"]],
        ["negative_number", "numeric", "condition"],
    ),
    (
        "防冻加热器的额定功率是多少千瓦？",
        "防冻加热器额定功率为1.5千瓦，通信柜电源额定功率为0.8千瓦。",
        [["1.5千瓦"]],
        ["numeric", "unit", "near_value"],
    ),
    (
        "主风速仪的采样间隔是多少秒？",
        "主风速仪每2秒采样一次，温湿度仪每30秒采样一次。",
        [["每2秒采样一次", "2秒"]],
        ["numeric", "near_value", "direct"],
    ),
    (
        "主雨量桶的口径是多少毫米？",
        "主雨量桶口径为200毫米，备用雨量桶口径为250毫米。",
        [["200毫米"]],
        ["numeric", "unit", "near_value"],
    ),
    (
        "雨量累计值每天何时归零？",
        "雨量累计值每日00:10归零，风速日报每日00:20生成。",
        [["每日00:10归零", "00:10"]],
        ["time", "near_value"],
    ),
    (
        "阵风达到什么条件分别触发红色告警和停塔作业？",
        "阵风超过15米/秒时触发红色告警，超过25米/秒时停止登塔作业。",
        [["超过15米/秒", "15米/秒"], ["超过25米/秒", "25米/秒"]],
        ["numeric", "multi_fact", "near_value"],
    ),
    (
        "温度探头与标准器的偏差达到什么条件必须校准？",
        "温度探头与标准器偏差超过0.6摄氏度须校准，偏差0.3摄氏度只登记。",
        [["超过0.6摄氏度", "0.6摄氏度"], ["须校准"]],
        ["numeric", "unit", "condition", "near_value"],
    ),
    (
        "数据上传失败到什么程度才转人工排查？",
        "连续3次上传失败须转人工排查，单次失败只进行自动重试。",
        [["连续3次上传失败"], ["转人工排查"]],
        ["numeric", "exception", "condition"],
    ),
    (
        "故障工单的首次响应时限是多少分钟？",
        "故障工单须在20分钟内首次响应，并在120分钟内提交处置记录。",
        [["20分钟内首次响应", "20分钟内"]],
        ["numeric", "near_value", "time"],
    ),
    (
        "原始观测数据和故障日志分别保留多久？",
        "原始观测数据保留90天，故障日志保留180天。",
        [["原始观测数据保留90天", "90天"], ["故障日志保留180天", "180天"]],
        ["numeric", "multi_fact", "near_value"],
    ),
    (
        "雷暴预警期间，停止采样后可以登塔吗？",
        "雷暴预警期间禁止登塔，即使设备已停止采样。",
        [["禁止登塔"]],
        ["negative", "exception", "condition"],
    ),
]

UNANSWERABLE = [
    ("主风速仪的制造商名称是什么？", "制造商名称"),
    ("岚峰主站的精确经纬度坐标是多少？", "经纬度坐标"),
    ("观测站每月电费是多少元？", "每月电费"),
]


def _write_corpus() -> None:
    """写入独立的 UTF-8 模拟业务资料，保留相近数值作为真实干扰。"""

    lines = [
        "岚峰气象观测站运维规则",
        "原创模拟业务资料；仅供 TraceRAG 的 006 整改后补充回归，不代表真实气象站制度。",
        "",
        "一、站点与设备",
    ]
    for index, (_, evidence, _, _) in enumerate(ANSWERABLE, start=1):
        if index == 5:
            lines += ["", "二、采样与告警"]
        if index == 9:
            lines += ["", "三、故障与安全"]
        lines.append(f"第{index:02d}条 {evidence}")
    (ROOT / SOURCE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_questions() -> None:
    """生成完整题目 schema，并把缺失信息写成明确的拒答题。"""

    cases = []
    for index, (query, evidence, fact_groups, tags) in enumerate(ANSWERABLE, start=1):
        cases.append({
            "id": f"B006-S-A{index:02d}",
            "query": query,
            "answerable": True,
            "expected_source": SOURCE,
            "expected_evidence": evidence,
            "expected_answer_facts": fact_groups,
            "allow_citations": True,
            "refusal_expected": False,
            "manual_review_condition": "核对问题对象、全部数值及单位、否定条件和对应引文；额外断言须人工复核。",
            "tags": tags,
        })
    for index, (query, _) in enumerate(UNANSWERABLE, start=1):
        cases.append({
            "id": f"B006-S-N{index:02d}",
            "query": query,
            "answerable": False,
            "expected_source": SOURCE,
            "expected_evidence": None,
            "expected_answer_facts": [],
            "allow_citations": False,
            "refusal_expected": True,
            "manual_review_condition": "资料未提供该字段；须明确拒答且没有引用。",
            "tags": ["unanswerable", "missing_field"],
        })
    assert len(cases) == 15 and len({case["id"] for case in cases}) == 15
    (ROOT / QUESTIONS).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_metadata() -> None:
    """记录补充回归的来源、用途、限制和两个冻结输入的 SHA-256。"""

    fingerprints = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (SOURCE, QUESTIONS)
    }
    metadata = {
        "dataset": "TraceRAG-006-supplement-r1",
        "created_on": "2026-10-05",
        "source_type": "原创模拟业务语料",
        "provenance": "为数字引用判定整改后的补充回归专门编写，无外部真实气象站资料来源。",
        "scope_limit": "15题补充回归不能替代已冻结的006完整50题独立验收，也不能代表真实业务准确率。",
        "corpus_files": [SOURCE],
        "questions": QUESTIONS,
        "answerable_count": 12,
        "unanswerable_count": 3,
        "cli_dataset_mode": "development",
        "not_full_independent_acceptance": True,
        "sha256": fingerprints,
    }
    (ROOT / "语料元数据.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """生成资料、题目和冻结指纹；有意修订时才再次执行。"""

    ROOT.mkdir(parents=True, exist_ok=True)
    _write_corpus()
    _write_questions()
    _write_metadata()
    print(f"补充回归样本已生成：{ROOT}")


if __name__ == "__main__":
    main()
