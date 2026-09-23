# TraceRAG

TraceRAG 是一个可溯源的知识库问答系统。V0.1 已形成从文档导入、向量检索到带来源引用回答的本地闭环。

系统仅根据检索到的证据作答；相关度不足、模型回答无法映射到有效来源时会拒答。

## V0.1 已实现

- 支持 PDF、Markdown、TXT 导入；PDF 保留页码，文本支持 UTF-8 与 GB18030。
- 生成可追溯 Chunk，使用 OpenAI 兼容 Embedding API 建立 FAISS 余弦相似度索引。
- 向量索引以不可变快照保存至本地，重启后自动加载；相同文档重复上传会跳过。
- 返回 Top-K 检索分数、回答引用与来源片段；低置信度问题会直接拒答。
- 提供 FastAPI 服务和 Streamlit 浏览器界面。

## 项目结构

```text
app/                  FastAPI 应用包
  main.py             服务入口与健康检查
  api.py              上传与查询 API
  runtime.py          文档入库、索引持久化和 RAG 运行时
  chat.py, rag.py     证据约束回答、引用映射与拒答
  ui.py               Streamlit 浏览器界面
tests/                自动化测试
data/raw/             原始文档（本地运行数据，不提交）
data/processed/       处理后数据（本地运行数据，不提交）
data/indexes/         向量索引（本地运行数据，不提交）
```

## 本地启动

Python 3.11 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple -e ".[dev]"
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

访问 <http://127.0.0.1:8000/health>，应返回健康状态。可在 `.env` 中调整 `TRACERAG_` 前缀的配置；`.env` 不会提交到版本库。

Embedding 和 Chat 服务需要 API 密钥：在 `.env` 中填写 `TRACERAG_API_KEY`，也可用已有的 `OPENAI_API_KEY` 环境变量。没有密钥时 `/health` 仍可访问，但上传和问答接口会返回 503。

另开一个 PowerShell 窗口启动浏览器界面：

```powershell
streamlit run app/ui.py --server.address 127.0.0.1
```

界面默认连接 `http://127.0.0.1:8000`，也可在侧栏更改 FastAPI 地址。当前 API 没有用户认证，请仅在可信的本机或内网使用，不要直接暴露到公网。

常用 `.env` 配置：`TRACERAG_BASE_URL`（兼容服务地址）、`TRACERAG_EMBEDDING_MODEL`、`TRACERAG_CHAT_MODEL`、`TRACERAG_INDEX_DIR`（本地索引目录）和 `TRACERAG_REJECT_THRESHOLD`（余弦相似度拒答阈值，默认 `0.25`）。

## 测试

```powershell
python -m pytest
```

## 当前接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | 返回服务、环境和版本信息 |
| POST | `/documents` | 上传 PDF、Markdown 或 TXT，解析并写入本地向量索引（最大 10 MiB） |
| POST | `/query` | 返回回答、拒答状态、引用来源和 Top-K 调试检索结果 |

## 当前限制

- PDF 仅提取已有文本层；扫描件 OCR、图片和复杂表格不在 V0.1 范围内。
- 拒答使用余弦相似度阈值，默认值为 `0.25`；应按所用 Embedding 模型和数据调整。
- 当前为单一本地索引，不含用户登录、知识库隔离、删除文档、BM25、Reranker 或评测框架。
- FastAPI 没有认证授权，默认仅绑定 `127.0.0.1`；不要在未加访问控制时暴露到公网。
