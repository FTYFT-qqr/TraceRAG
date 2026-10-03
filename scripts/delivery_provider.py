"""提供仅用于离线工程验收的固定 OpenAI 兼容服务，不代表真实模型能力。"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException


app = FastAPI(title="TraceRAG 固定验收协议服务")


@app.post("/v1/embeddings")
def embeddings(payload: dict) -> dict:
    """返回固定二维向量，只检查协议、索引保存和恢复，不测语义质量。"""
    texts = payload["input"]
    if isinstance(texts, str):
        texts = [texts]
    return {
        "object": "list",
        "model": payload["model"],
        "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(texts)],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
def chat(payload: dict) -> dict:
    """针对固定演示问题选择原文编号，未知问题明确拒答。"""
    request = json.loads(payload["messages"][-1]["content"])
    question = request["question"]
    ids = []
    if question == "工程演示设备最长可以借用多久？":
        for source in request["evidence"]:
            for passage in source["passages"]:
                if "最长4小时" in passage["text"]:
                    ids.append(passage["id"])
                    break
            if ids:
                break
    elif question != "工程演示设备的购买价格是多少？":
        raise HTTPException(status_code=400, detail="固定协议服务只支持工程演示题目。")
    content = json.dumps({"refused": not ids, "passage_ids": ids}, ensure_ascii=False)
    return {
        "id": "delivery-fixed-response", "object": "chat.completion", "created": 0,
        "model": payload["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
