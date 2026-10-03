"""验证原文短句选择、逐句引用及格式错误处理，不调用外部模型。"""

import json

import pytest

from app.grounded_answer import REFUSAL_ANSWER, evidence_passages, render_grounded_answer


def _payload(ids: list) -> str:
    """构造模型的短句编号选择响应，便于独立检验约束。"""

    return json.dumps({"refused": False, "passage_ids": ids})


def test_complete_passages_preserve_polarity_numbers_and_conditions() -> None:
    """模型无法改写短句或截掉数值、否定词与同句条件，每句均绑定原始来源。"""

    source = "校准期间设备不可借用；借用最长4.5小时。缺少配件时不能完成登记。"
    passages = evidence_passages({2: source})
    assert passages == {"2:1": (2, "校准期间设备不可借用；借用最长4.5小时"), "2:2": (2, "缺少配件时不能完成登记")}
    answer = render_grounded_answer(_payload(["2:1", "2:2", "2:1"]), passages)
    assert answer == "校准期间设备不可借用[2]。\n借用最长4.5小时[2]。\n缺少配件时不能完成登记[2]。"


def test_procedure_prerequisite_and_confirmation_stay_in_one_selectable_passage() -> None:
    """分号连接的预约必填信息与确认条件保留在同一选择单元中。"""

    source = "预约时填写姓名、课题编号和预计归还时间；收到柜台确认后才算成功。"
    passages = evidence_passages({1: source})
    assert len(passages) == 1
    answer = render_grounded_answer(_payload(["1:1"]), passages)
    assert "填写姓名、课题编号和预计归还时间[1]" in answer
    assert "收到柜台确认后才算成功[1]" in answer


def test_refusal_requires_empty_passage_selection() -> None:
    """明确拒答只能使用空选择，不能隐藏引用或混入事实。"""

    assert render_grounded_answer('{"refused": true, "passage_ids": []}', {}) == REFUSAL_ANSWER
    with pytest.raises(ValueError):
        render_grounded_answer('{"refused": true, "passage_ids": ["1:1"]}', {})


@pytest.mark.parametrize("ids", [["2:1"], ["1:0"], ["1:99"], ["1:1", "fake"], [True], [1], [None], [{}], []])
def test_invalid_selections_are_not_rendered(ids: list) -> None:
    """伪造编号、错误类型和空选择不能产生正式回答。"""

    with pytest.raises(ValueError):
        render_grounded_answer(_payload(ids), evidence_passages({1: "设备不可借用。"}))


@pytest.mark.parametrize("content", [
    "借用最长4小时[1]。",
    "[]",
    '{}',
    '{"refused": false, "passage_ids": []}',
    '{"refused": "false", "passage_ids": []}',
    '{"refused": false, "passage_ids": "1:1"}',
    '{"refused": false, "passage_ids": ["1:1"], "answer": "自由文本"}',
    '{"refused": false, "claims": [{"quote": "可借用", "source_ids": [1]}]}',
])
def test_invalid_output_contract_is_reported_as_error(content: str) -> None:
    """错误格式明确报错，不降级输出未经校验的自由回答或伪装成正常拒答。"""

    with pytest.raises(ValueError):
        render_grounded_answer(content, evidence_passages({1: "设备不可借用。"}))


def test_original_reference_labels_do_not_enter_generated_citation_space() -> None:
    """原文参考编号保留显示但不被RAG当作本次生成的引用。"""

    passages = evidence_passages({1: "规则见[7]。"})
    assert render_grounded_answer(_payload(["1:1"]), passages) == "规则见［7］[1]。"


def test_cross_reference_can_select_both_original_premises() -> None:
    """两个不同来源短句可以共同说明间接地点，编号由代码确定。"""

    passages = evidence_passages({1: "纸质报告领取地点与年度体检相同。", 2: "年度体检报告可在一楼登记台领取。"})
    answer = render_grounded_answer(_payload(["1:1", "2:1"]), passages)
    assert answer == "纸质报告领取地点与年度体检相同[1]。\n年度体检报告可在一楼登记台领取[2]。"
