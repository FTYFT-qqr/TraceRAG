"""生成 006 原创模拟业务语料、互不重叠的题集与输入指纹。"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TXT_NAME = "云栖工坊订单服务.txt"
MD_NAME = "远舟冷链交接手册.md"
PDF_NAME = "青岚园区能源服务规则.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# 各文档前四条专供开发集；其余条款专供最终验收集。
# 每条保存独立业务事实，避免同一事实换种问法凑题数。
TXT = [
    ("工坊试制订单由哪个窗口受理？", "试制订单统一由一号服务窗口受理。", ["一号服务窗口"], ["direct"]),
    ("试制订单需要提前几天提交？", "试制订单须提前3个工作日提交。", ["提前3个工作日"], ["numeric"]),
    ("工坊接受电话口头改图吗？", "图纸变更不接受电话口头通知。", ["不接受电话口头通知"], ["negative"]),
    ("订单样件编号由谁分配？", "样件编号由计划专员在确认订单后分配。", ["计划专员"], ["direct"]),
    ("普通试制单的最小批量是多少件？", "普通试制单的最小批量为12件。", ["12件"], ["numeric"]),
    ("耐候试制件需要什么包装？", "耐候试制件须使用防潮袋单件封装。", ["防潮袋单件封装"], ["direct"]),
    ("加急订单的截止受理时刻是什么？", "加急订单仅在工作日15点前受理。", ["工作日15点前"], ["exception", "time"]),
    ("首件确认由哪个岗位签字？", "首件确认单须由质量工程师签字。", ["质量工程师签字"], ["direct"]),
    ("批量开工前要先完成什么记录？", "批量开工前必须完成设备点检记录。", ["设备点检记录"], ["direct"]),
    ("材料替代申请交给谁审批？", "材料替代申请由工艺主管审批。", ["工艺主管审批"], ["direct"]),
    ("超过多少毫米的尺寸偏差需要复测？", "尺寸偏差超过0.5毫米时必须复测。", ["超过0.5毫米"], ["numeric", "condition"]),
    ("不合格样件应放在哪里？", "不合格样件应放入红色隔离箱。", ["红色隔离箱"], ["direct"]),
    ("客户自带材料在入库前需要做什么？", "客户自带材料入库前须拍照登记。", ["拍照登记"], ["condition"]),
    ("已经开始加工的订单能否免费取消？", "已开始加工的订单不支持免费取消。", ["不支持免费取消"], ["negative", "condition"]),
    ("成品出库由谁复核数量？", "成品出库数量由仓储复核员核对。", ["仓储复核员"], ["direct"]),
    ("试制报告通过什么方式交付？", "试制报告通过客户专用邮箱交付。", ["客户专用邮箱"], ["direct"]),
    ("客户对尺寸复测结果有异议时可以做什么？", "客户可在收件后2个工作日内申请尺寸复核。", ["2个工作日内", "申请尺寸复核"], ["numeric", "condition"]),
    ("工坊的逾期领取物品如何处置？", "逾期30天未领取的成品转入暂存架。", ["逾期30天", "转入暂存架"], ["numeric", "condition"]),
]

MD = [
    ("冷链车辆出车前应核对什么标签？", "出车前司机须核对车厢温区标签。", ["车厢温区标签"], ["direct"]),
    ("交接单由哪个岗位创建？", "交接单由发货调度员创建。", ["发货调度员"], ["direct"]),
    ("温度记录仪电量不足时能否发车？", "记录仪电量低于20%时不得发车。", ["低于20%", "不得发车"], ["negative", "numeric"]),
    ("到站照片要保存几张？", "到站时须保存2张封签照片。", ["2张封签照片"], ["numeric"]),
    ("鲜品装车前的车厢预冷目标是多少度？", "鲜品装车前车厢须预冷至4摄氏度。", ["4摄氏度"], ["numeric"]),
    ("冻品装车采用哪个温区？", "冻品装车使用负18摄氏度温区。", ["负18摄氏度"], ["numeric", "near_name"]),
    ("交接时封签号码在哪里登记？", "封签号码必须登记在电子交接单。", ["电子交接单"], ["direct"]),
    ("司机发现封签破损应先联系谁？", "发现封签破损时司机须先联系值班调度。", ["值班调度"], ["condition"]),
    ("卸货等待超过多久要记录异常？", "卸货等待超过45分钟须记录异常原因。", ["超过45分钟", "记录异常原因"], ["numeric", "condition"]),
    ("鲜品到站温度高于多少度要隔离？", "鲜品到站温度高于8摄氏度须先隔离。", ["高于8摄氏度", "先隔离"], ["numeric", "condition"]),
    ("冻品开箱抽查由谁共同见证？", "冻品开箱抽查须由收货员与司机共同见证。", ["收货员与司机共同见证"], ["direct"]),
    ("夜间交接允许先签字后查温吗？", "夜间交接不得先签字后查验温度。", ["不得先签字后查验温度"], ["negative"]),
    ("临时改道由谁批准？", "临时改道须获得运输主管批准。", ["运输主管批准"], ["direct"]),
    ("车辆故障时备车须在多久内到场？", "车辆故障后备车应在90分钟内到场。", ["90分钟内"], ["numeric"]),
    ("异常温度曲线保存在哪里？", "异常温度曲线须上传至批次追溯档案。", ["批次追溯档案"], ["direct"]),
    ("拒收货物的暂存标识是什么颜色？", "拒收货物须贴紫色暂存标识。", ["紫色暂存标识"], ["near_name"]),
    ("返程空箱需要在何时消毒？", "返程空箱须在次日首车出发前消毒。", ["次日首车出发前消毒"], ["time", "condition"]),
    ("电子交接单的归档期限是多久？", "电子交接单须保留18个月。", ["保留18个月"], ["numeric"]),
]

PDF = [
    ("园区能源抄表由哪个部门负责？", "园区月度抄表由能源服务部负责。", ["能源服务部"], ["direct"]),
    ("月度用能通知一般在每月几号发布？", "月度用能通知于每月5日发布。", ["每月5日"], ["time"]),
    ("企业申请电表更名需要什么材料？", "电表更名申请须附营业执照复印件。", ["营业执照复印件"], ["direct"]),
    ("物业可以代企业签署用能确认书吗？", "物业不得代企业签署用能确认书。", ["不得代企业签署"], ["negative"]),
    ("企业发现抄表数值异常应在几天内申诉？", "抄表异议须在通知后7个自然日内提出。", ["7个自然日内"], ["numeric", "condition"]),
    ("园区电费发票从哪个入口下载？", "电费发票可在企业服务门户下载。", ["企业服务门户"], ["direct"]),
    ("公共照明常规关闭时间是几点？", "公共照明常规于23点关闭。", ["23点关闭"], ["time"]),
    ("夜间施工申请需要提前多久提交？", "夜间施工用电申请须提前48小时提交。", ["提前48小时"], ["numeric", "condition"]),
    ("夜间施工用电由谁审批？", "夜间施工用电由园区安全主管审批。", ["园区安全主管"], ["near_name"]),
    ("配电柜周边多少米内不得堆放物品？", "配电柜周边1.5米内不得堆放物品。", ["1.5米内", "不得堆放物品"], ["numeric", "negative"]),
    ("园区计划停电至少提前多久告知？", "计划停电至少提前72小时告知企业。", ["提前72小时"], ["numeric"]),
    ("突发停电应从哪个号码报修？", "突发停电应拨打园区报修短号806。", ["报修短号806"], ["numeric"]),
    ("充电车位每次连续占用上限是多少？", "充电车位单次连续占用上限为4小时。", ["4小时"], ["numeric"]),
    ("消防通道的充电插座可以使用吗？", "消防通道插座禁止用于车辆充电。", ["禁止用于车辆充电"], ["negative"]),
    ("漏水报警由谁先到现场确认？", "漏水报警由值班水务员先到现场确认。", ["值班水务员"], ["near_name"]),
    ("每季度节能建议提交到哪里？", "季度节能建议提交至能源服务部邮箱。", ["能源服务部邮箱"], ["direct"]),
    ("新增大功率设备的申报阈值是多少？", "新增功率超过30千瓦的设备须事先申报。", ["超过30千瓦", "事先申报"], ["numeric", "condition"]),
    ("非工作时间的故障工单由谁接收？", "非工作时间故障工单由夜班值守员接收。", ["夜班值守员"], ["time"]),
]

# 无答案题选择各文档均没有给出的具体字段；不从另一个来源借用答案。
UNANSWERABLE = [
    (TXT_NAME, "工坊试制件的国际快递运费是多少？", ["unanswerable", "fee"]),
    (TXT_NAME, "工坊负责人姓名和手机号码是什么？", ["unanswerable", "identity"]),
    (TXT_NAME, "普通试制单可以使用哪张信用卡付款？", ["unanswerable", "payment"]),
    (TXT_NAME, "成品暂存架的精确货位编号是什么？", ["unanswerable", "location"]),
    (MD_NAME, "远舟冷链的车牌号码是什么？", ["unanswerable", "identity"]),
    (MD_NAME, "冻品运输的保险金额是多少元？", ["unanswerable", "amount"]),
    (MD_NAME, "值班调度的手机号码是多少？", ["unanswerable", "contact"]),
    (PDF_NAME, "青岚园区商业电价每千瓦时是多少元？", ["unanswerable", "fee"]),
    (PDF_NAME, "园区安全主管的姓名是什么？", ["unanswerable", "identity"]),
    (PDF_NAME, "企业服务门户的网址是什么？", ["unanswerable", "url"]),
]

TXT_CONTEXT = {
    6: (
        "受理与准备说明：试制订单会经过受理、信息核对、计划确认和材料准备等环节。"
        "接到客户询问时，办理人员应先核对订单记录所处的环节，再回到具体条款查找相应要求。"
        "样件编号、工艺文件和客户提交的材料说明属于不同记录，不能把其中一项当作另一项的替代。"
        "样件进入加工前，岗位之间要交接已经确认的信息；若信息仍有疑问，应继续查明原始记录。"
        "这一段用于解释流程背景，具体数量、时间和签字要求仍以各条款为准。"
    ),
    12: (
        "加工与交付说明：工坊按订单记录跟踪加工节点，并在质量确认之后准备出库资料。"
        "现场记录、客户通知和最终交付是不同动作，不能根据其中一个动作推断其他动作已经完成。"
        "面对尺寸、材料或交付方面的询问，应先辨认问题对应的条款，再核实记录中的具体对象。"
        "相近名称的岗位各自承担不同任务，办理人员需要在答复中保持岗位和条件的一致。"
        "这一段没有增加新的时限或费用标准，具体处置以编号条款为准。"
    ),
}

MD_CONTEXT = {
    6: (
        "发运背景：车辆出库前的核对工作涉及温区、记录仪、封签和交接文件。"
        "这些对象虽然会出现在同一批次的资料中，却代表不同的操作步骤。"
        "司机应根据当前步骤核对对应记录，调度人员也需要区分出车前确认和到站后确认。"
        "阅读本手册时，不能将鲜品与冻品的温区要求混为一谈，也不能用一般情况推定异常处置。"
        "具体温度、阈值和岗位由后续相应条款逐项给出，本段只解释资料结构。"
    ),
    12: (
        "异常背景：到站交接后还可能涉及临时路线、车辆故障、货物拒收和资料归档。"
        "这些情形的触发条件与处理记录各不相同，不能凭某一项记录推断所有后续动作。"
        "如果需要回答具体期限、接收岗位或暂存标识，应核对所问情形及相应条款。"
        "电子记录和实物标识承担不同作用；手册将它们分开叙述，以便独立追踪每一步。"
        "本段作为流程背景，不新增任何温度或保险方面的约定。"
    ),
}


def _write_text_corpus() -> None:
    """按固定顺序写入两份 UTF-8 纯文本业务资料。"""

    txt_lines = ["云栖工坊订单服务规则", "原创模拟业务资料；仅供 TraceRAG 泛化评测，不代表真实工坊政策。", ""]
    for index, (_, evidence, _, _) in enumerate(TXT, start=1):
        txt_lines.append(f"第{index:02d}条 {evidence}")
        if index in TXT_CONTEXT:
            txt_lines += ["", TXT_CONTEXT[index], ""]
    (ROOT / TXT_NAME).write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    md_lines = ["# 远舟冷链交接手册", "", "> 原创模拟业务资料；仅供 TraceRAG 泛化评测，不代表真实企业流程。", ""]
    for index, (_, evidence, _, _) in enumerate(MD, start=1):
        if index in {1, 7, 13}:
            md_lines += [f"## {('发运准备', '到站交接', '异常与归档')[(index - 1) // 6]}", ""]
        md_lines += [f"- 条款 {index:02d}：{evidence}", ""]
        if index in MD_CONTEXT:
            md_lines += [MD_CONTEXT[index], ""]
    (ROOT / MD_NAME).write_text("\n".join(md_lines), encoding="utf-8")


def _write_pdf_corpus() -> None:
    """经 Edge 打印固定三页 PDF，保留可由正式 Loader 抽取的中文文本层。"""

    if not EDGE.is_file():
        raise RuntimeError("找不到 Microsoft Edge，无法生成含中文文本层的 PDF。")
    pages = []
    for page_number in range(3):
        start = page_number * 6
        clauses = "\n".join(
            f"<p>条款 {index + 1:02d}：{html.escape(PDF[index][1])}</p>"
            for index in range(start, start + 6)
        )
        pages.append(
            f'<section class="sheet"><h1>青岚园区能源服务规则</h1>'
            f'<p class="intro">原创模拟业务资料 · 第 {page_number + 1} 页 / 共 3 页</p>'
            f"{clauses}</section>"
        )
    page_html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<style>@page{size:A4;margin:12mm}body{font-family:"Microsoft YaHei",sans-serif;'
        'font-size:12pt;color:#111}.sheet{break-after:page;page-break-after:always}'
        '.sheet:last-child{break-after:auto;page-break-after:auto}h1{font-size:19pt}'
        '.intro{font-size:10pt;color:#555;margin-bottom:24px}p{margin:15px 0}</style>'
        '<body>' + "\n".join(pages) + "</body></html>"
    )
    with tempfile.TemporaryDirectory(prefix="tracerag-006-pdf-") as scratch:
        scratch_path = Path(scratch)
        html_path = scratch_path / "content.html"
        html_path.write_text(page_html, encoding="utf-8")
        output = ROOT / PDF_NAME
        completed = subprocess.run(
            [str(EDGE), "--headless", "--disable-gpu", "--no-first-run",
             "--no-pdf-header-footer", f"--user-data-dir={scratch_path / 'profile'}",
             f"--print-to-pdf={output}", html_path.as_uri()],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"Edge PDF 生成失败：{completed.returncode} {completed.stderr[-500:]}")


def _case(case_id: str, source: str, record: tuple, *, page: int | None = None) -> dict:
    """把单个条款转换成既有质量判定器识别的完整题目。"""

    query, evidence, facts, tags = record
    result = {
        "id": case_id,
        "query": query,
        "answerable": True,
        "expected_source": source,
        "expected_evidence": evidence,
        "expected_answer_facts": [[fact] for fact in facts],
        "allow_citations": True,
        "refusal_expected": False,
        "manual_review_condition": "核对全部预期事实、证据原文和引用来源；额外断言须人工复核。",
        "tags": tags,
    }
    if page is not None:
        result["expected_page_number"] = page
    return result


def _build_questions() -> None:
    """以互不重叠的条款构造开发集、40道答案题和10道拒答题。"""

    collections = [(TXT_NAME, TXT), (MD_NAME, MD), (PDF_NAME, PDF)]
    dev, acceptance = [], []
    for source_index, (source, records) in enumerate(collections, start=1):
        for record_index, record in enumerate(records[:4], start=1):
            page = (record_index - 1) // 6 + 1 if source == PDF_NAME else None
            dev.append(_case(f"B006-D{source_index}-{record_index:02d}", source, record, page=page))

    for source_index, (source, records) in enumerate(collections, start=1):
        singles = records[4:] if source == PDF_NAME else records[5:17]
        for record_index, record in enumerate(singles, start=5 if source == PDF_NAME else 6):
            page = (record_index - 1) // 6 + 1 if source == PDF_NAME else None
            acceptance.append(_case(f"B006-A{source_index}-{record_index:02d}", source, record, page=page))
        if source != PDF_NAME:
            first, second = records[4], records[17]
            composite_query = (
                "普通试制单的最小批量是多少，逾期成品如何处置？"
                if source == TXT_NAME else
                "鲜品装车前预冷目标是多少，电子交接单须保存多久？"
            )
            composite = _case(f"B006-A{source_index}-05", source, first)
            composite["query"] = composite_query
            composite["expected_evidence"] = [first[1], second[1]]
            composite["expected_answer_facts"] = [[fact] for fact in first[2] + second[2]]
            composite["tags"] = ["multi_segment", "multi_fact", "numeric"]
            acceptance.append(composite)

    for index, (source, query, tags) in enumerate(UNANSWERABLE, start=1):
        acceptance.append({
            "id": f"B006-N{index:02d}",
            "query": query,
            "answerable": False,
            "expected_source": source,
            "expected_evidence": None,
            "expected_answer_facts": [],
            "allow_citations": False,
            "refusal_expected": True,
            "manual_review_condition": "资料未给出该信息；必须明确拒答且不得附引用。",
            "tags": tags,
        })
    assert len(dev) == 12
    assert len(acceptance) == 50
    assert sum(case["answerable"] for case in acceptance) == 40
    assert len({case["id"] for case in dev + acceptance}) == 62
    (ROOT / "questions_development.json").write_text(
        json.dumps(dev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "questions_acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_metadata() -> None:
    """冻结五个输入文件的 SHA-256，并明确模拟来源及题集分工。"""

    files = [TXT_NAME, MD_NAME, PDF_NAME, "questions_development.json", "questions_acceptance.json"]
    fingerprints = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in files
    }
    metadata = {
        "dataset": "TraceRAG-006",
        "created_on": "2026-10-05",
        "source_type": "原创模拟业务语料",
        "provenance": "为评测专门编写，无外部企业、政府或园区资料来源；不得当作真实业务规则引用。",
        "scope_limit": "只能验证系统对本组模拟政策文本的检索和回答，不能代表真实业务分布。",
        "corpus_files": [TXT_NAME, MD_NAME, PDF_NAME],
        "development_questions": "questions_development.json",
        "acceptance_questions": "questions_acceptance.json",
        "development_count": 12,
        "acceptance_answerable_count": 40,
        "acceptance_unanswerable_count": 10,
        "pdf_page_number_basis": "1-based，与 app.document_loader.load_document 一致",
        "sha256": fingerprints,
    }
    (ROOT / "语料元数据.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """依次生成语料、题集和元数据，保持每次构建的内部一致性。"""

    ROOT.mkdir(parents=True, exist_ok=True)
    _write_text_corpus()
    _write_pdf_corpus()
    _build_questions()
    _write_metadata()
    print(f"006 语料与题集已生成：{ROOT}")


if __name__ == "__main__":
    main()
