# TraceRAG

TraceRAG 是一个可溯源的企业知识库问答系统。本仓库当前完成 **V0.1 阶段 1：项目骨架**：基础目录、环境配置、日志和 FastAPI 健康检查。

后续阶段会依次加入文档导入、文本切分、Embedding 与 FAISS、检索、RAG 引用与拒答，以及简易界面。当前不会提前引入这些能力。

## 项目结构

```text
app/                 FastAPI 应用和基础配置
tests/               自动化测试
data/raw/            原始文档（本地运行数据，不提交）
data/processed/      处理后数据（本地运行数据，不提交）
data/indexes/        向量索引（本地运行数据，不提交）
```

## 本地启动

Python 3.11 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

访问 <http://127.0.0.1:8000/health>，应返回健康状态。可在 `.env` 中调整 `TRACERAG_` 前缀的配置；`.env` 不会提交到版本库。

## 测试

```powershell
python -m pytest
```

## 当前接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | 返回服务、环境和版本信息 |
