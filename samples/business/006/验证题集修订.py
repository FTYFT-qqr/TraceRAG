"""验证 006 v2 题集仅改变指定问法且仍满足原文和切分约束。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from app.chunking import chunk_pages  # noqa: E402
from app.document_loader import load_document  # noqa: E402
from scripts.evaluate_v02 import load_cases, validate_cases_against_corpus  # noqa: E402


def _sha256(path: Path) -> str:
    """读取文件原始字节计算 SHA-256，避免文本换行转换影响指纹。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidences(case: dict) -> list[str]:
    """统一单证据和多证据题目的原文列表。"""

    raw = case["expected_evidence"]
    return [raw] if isinstance(raw, str) else list(raw or [])


def main() -> None:
    """核对新旧题集差异、正式 Loader 来源与默认切分中的完整证据。"""

    old_metadata = json.loads((ROOT / "语料元数据.json").read_text(encoding="utf-8"))
    new_metadata = json.loads((ROOT / "语料元数据_v2.json").read_text(encoding="utf-8"))
    old_name = "questions_acceptance.json"
    new_name = new_metadata["acceptance_questions"]
    assert _sha256(ROOT / old_name) == old_metadata["sha256"][old_name]
    assert new_metadata["source_v1_question_sha256"] == old_metadata["sha256"][old_name]
    for name, expected in new_metadata["sha256"].items():
        assert _sha256(ROOT / name) == expected, f"v2 输入指纹变化：{name}"

    old_cases = load_cases(ROOT / old_name, dataset_mode="acceptance")
    new_cases = load_cases(ROOT / new_name, dataset_mode="acceptance")
    assert len(old_cases) == len(new_cases) == 50
    differences = []
    for old, new in zip(old_cases, new_cases, strict=True):
        assert old["id"] == new["id"]
        for field in old.keys() | new.keys():
            if old.get(field) != new.get(field):
                differences.append((old["id"], field, old.get(field), new.get(field)))
    assert differences == [
        ("B006-A1-17", "query", new_metadata["original_query"], new_metadata["revised_query"])
    ], f"v2 出现额外修改：{differences}"

    validate_cases_against_corpus(new_cases, ROOT)
    chunks = {}
    for source in new_metadata["corpus_files"]:
        pages = load_document(source, (ROOT / source).read_bytes())
        assert pages and all(page.content.strip() for page in pages)
        chunks[source] = chunk_pages(pages, chunk_size=700, overlap=100)
    for case in new_cases:
        if not case["answerable"]:
            continue
        for evidence in _evidences(case):
            assert any(
                chunk.file_name == case["expected_source"]
                and chunk.page_number == case.get("expected_page_number")
                and evidence in chunk.content
                for chunk in chunks[case["expected_source"]]
            ), f"v2 证据不在默认切分片段中：{case['id']}"
    print(json.dumps({
        "old_question_sha256": _sha256(ROOT / old_name),
        "new_question_sha256": _sha256(ROOT / new_name),
        "changed_case_id": "B006-A1-17",
        "changed_field": "query",
        "total_cases": 50,
        "answerable_cases": sum(case["answerable"] for case in new_cases),
        "unanswerable_cases": sum(not case["answerable"] for case in new_cases),
        "loader_and_default_chunk_validation": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
