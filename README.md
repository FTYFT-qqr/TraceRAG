# TraceRAG

TraceRAG 是一个可溯源的知识库问答系统。当前包版本为 **0.2.0**，包含文档导入、向量/BM25/RRF 检索、原文回答、逐句引用与离线评测。

系统仅根据检索到的证据作答；相关度不足、模型回答无法映射到有效来源时会拒答。

## 已实现功能

- 支持 PDF、Markdown、TXT 导入；PDF 保留页码，文本支持 UTF-8 与 GB18030。
- 生成可追溯 Chunk，使用 OpenAI 兼容 API 或配置的本地模型建立 FAISS 余弦相似度索引。
- 向量快照记录 Embedding provider、模型身份、远程服务端点或本地模型内容指纹及向量维度；同名更新替换旧内容，相同文档重复上传会跳过。
- Embedding 身份不匹配时拒绝复用已有索引；身份不完整的旧快照会用当前模型重建向量并保留旧 generation。
- 返回 Top-K 检索分数、回答引用与来源片段；低置信度问题会直接拒答。
- Top-K 是最终交给模型参考并在界面展示的片段上限；是否记载问题可内部扩展候选、优先保留同主题的“未列出”证据，但最终数量不超过所选上限。
- 提供 FastAPI 服务和 Streamlit 浏览器界面。
- 在线支持 BM25 与 RRF 混合召回；模型选择原文短句编号，代码生成文本与逐句引用。

## 完整版交付入口

| 内容 | 入口 |
| --- | --- |
| Docker API/UI 启动、本地模型挂载与索引持久化 | [容器启动说明](docs/整理记录/版本0.2/工程交付/容器启动说明.md) |
| 在线链路与离线评测架构 | [系统架构](docs/整理记录/版本0.2/工程交付/系统架构.md) |
| 固定演示、检索改进与简历表述 | [演示操作与项目展示](docs/整理记录/版本0.2/工程交付/演示操作与项目展示.md) |
| 独立环境测试与剩余交付项 | [交付验收记录](docs/整理记录/版本0.2/工程交付/交付验收记录.md) |
| 版本范围与升级说明 | [0.2.0 发布说明](docs/项目规划/版本0.2/发布说明.md) |

一键容器启动：复制 `.env.docker.example` 为 `.env.docker` 并填写服务配置，然后执行 `docker compose up -d --build --wait`。访问 `http://127.0.0.1:18501/`，API 地址为 `http://127.0.0.1:18000/`。使用本地模型时增加 `-f compose.yaml -f compose.local.yaml`，详细命令见容器说明。

## 项目结构

```text
app/                  FastAPI 应用包
  main.py             服务入口与健康检查
  api.py              上传与查询 API
  runtime.py          文档入库、索引持久化和 RAG 运行时
  chat.py, rag.py     证据约束回答、引用映射与拒答
  ui.py               Streamlit 浏览器界面
tests/                自动化测试
docs/                 按用途分类的中文文档
  文档导航.md         统一入口与存放规则
  项目规划/           开发路线、版本规划与待办
  整理记录/           整改过程、人工复核与验收结论
  评测数据/           原始结果、逐题报告与自动摘要
indexes/              本地索引快照（运行数据，不提交）
```

### 知识库数据保存方式

当前实现收到上传文件后，在内存中解析和切分；上传的原文件不会被应用另存一份。切分后的文本及来源信息保存在索引快照的 `chunks.json`，向量保存在 `index.faiss`，模型身份等信息保存在 `manifest.json`，`CURRENT` 指向当前快照。默认目录为 `indexes/default/`，可用 `TRACERAG_INDEX_DIR` 修改。这些都是本地文件，不依赖外部数据库；后续若需要多用户、大规模检索或更复杂的筛选，可以再改用向量数据库。

## 本地启动

包声明支持 Python 3.11 或更高版本；本轮交付实际验证 Python 3.13，以下依赖约束及 Docker 也使用 3.13：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple -c requirements-release.txt -e ".[dev]"
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

访问 <http://127.0.0.1:8000/health>，应返回健康状态。可在 `.env` 中调整 `TRACERAG_` 前缀的配置；`.env` 不会提交到版本库。

Chat 服务需要 API 密钥：在 `.env` 中填写 `TRACERAG_API_KEY`，也可用已有的 `OPENAI_API_KEY` 环境变量。Embedding 默认使用 OpenAI 兼容 API；也可以切换为本地 Sentence Transformers 模型。没有 Chat API 密钥时 `/health` 仍可访问，但上传和问答接口会返回 503。

### 使用本地 BGE-M3

模型权重放在 `D:/rag/embedding`，项目通过配置读取该目录，不会把约 2.27 GB 的权重复制进 Git 仓库。安装本地推理依赖：

```powershell
python -m pip install -e ".[local-embeddings]"
```

在项目根目录的 `.env` 中设置：

```dotenv
TRACERAG_EMBEDDING_PROVIDER=local
TRACERAG_LOCAL_EMBEDDING_PATH=D:/rag/embedding
TRACERAG_INDEX_DIR=indexes/bge-m3
```

`TRACERAG_EMBEDDING_DEVICE` 可设为 `cpu` 或 `cuda`；不设置时由 Sentence Transformers 自动选择。首次上传文档会用 BGE-M3 建立新的 1024 维索引；需要重新上传文档，之后的查询才会使用本地向量。保留 `TRACERAG_API_KEY` 和 `TRACERAG_BASE_URL` 供 Chat 服务使用。切换 Embedding 模型时始终使用单独的索引目录。

如果 Chat 使用 DeepSeek，在 `.env` 中设置 `TRACERAG_BASE_URL=https://api.deepseek.com`、`TRACERAG_CHAT_MODEL=deepseek-flash`，并将 `TRACERAG_API_KEY` 设为 DeepSeek API 密钥。需要代理时，另设 `TRACERAG_API_PROXY_URL`，例如 `http://127.0.0.1:7897`；此配置只用于远程 Chat 和 Embedding 请求。

另开一个 PowerShell 窗口启动浏览器界面：

```powershell
.\scripts\start-ui.ps1
```

此脚本会将 Streamlit 的运行配置目录放在系统临时目录，避免受限环境下无法写入用户目录而造成浏览器连接中断。

界面默认连接 `http://127.0.0.1:8000`，也可在侧栏更改 FastAPI 地址。当前 API 没有用户认证，请仅在可信的本机或内网使用，不要直接暴露到公网。

常用 `.env` 配置：`TRACERAG_BASE_URL`（Chat 兼容服务地址）、`TRACERAG_API_PROXY_URL`（可选远程 API 代理）、`TRACERAG_EMBEDDING_PROVIDER`（`openai` 或 `local`）、`TRACERAG_EMBEDDING_MODEL`（兼容 API 模型名）、`TRACERAG_LOCAL_EMBEDDING_PATH`（本地模型目录）、`TRACERAG_CHAT_MODEL`、`TRACERAG_INDEX_DIR`（本地索引目录）和 `TRACERAG_REJECT_THRESHOLD`（余弦相似度拒答阈值，默认 `0.25`）。

## 测试

```powershell
python -m pytest
```

已验收的 91 个基础用例均通过；真实样本验收为 14/15 条证据进入 Top-5、15/15 条有答案质量通过、4/4 条正确拒答。完整结果见 [基础测试记录](docs/整理记录/版本0.1/验收记录/基础测试记录.md) 和 [检索验收记录](docs/评测数据/版本0.1/验收记录/检索验收记录.md)。

用当前配置的真实 Embedding 和 Chat 模型运行 V0.1 最小检索验收：

```powershell
python scripts/evaluate_retrieval.py
```

当前 V0.1 真实模型验收达到 14/15 条有答案问题的 Top-5 证据命中，15/15 条有答案问题质量通过，4/4 条无答案问题正确拒答；待人工复核和执行错误均为 0。存在性问题可能使用扩大的候选集，Top-5 指标仍按实际前五名计算。

2026-09-28 V0.1 已完成冻结：规划 003 与验收标准 004 的拒答、回答质量和 Top-K 整改已通过，91/91 基础测试、真实样本验收及 LLM 证据约束专项均通过。后续工作进入独立的 V0.2 规划，详见 [V0.1 专项验收 005](docs/整理记录/版本0.1/整改记录/05_证据约束专项验收.md)、[V0.2 评测基线规划 001](docs/项目规划/版本0.2/版本规划.md) 和 [文档索引](docs/文档导航.md)。

脚本只使用 `samples/acceptance/` 中的项目样例，在内存中建立隔离索引，并把逐题结果写入 `docs/评测数据/版本0.1/验收记录/检索验收记录.md`；问题和样例片段会发送给 `.env` 中配置的 Chat 服务，不读取用户上传目录或写入应用索引。

## V0.2 离线检索对照

V0.2 提供固定 50 题的离线检索评测，可比较切分长度、Embedding 推理批次、BM25 和 BM25+RRF。运行 `python scripts/evaluate_v02.py --suite --output-dir docs/评测数据/版本0.2/临时运行` 会建立独立内存索引并输出逐题完整排名，不会修改应用使用的索引。默认只做本地检索；增加 `--quality` 才会把评测问题和检索到的样例片段发送到 `.env` 配置的 Chat 服务。

当前对照记录见 [V0.2 结果摘要](docs/评测数据/版本0.2/第一轮检索/对照摘要.md) 和 [V0.2 评测目录](docs/评测数据/版本0.2/报告导航.md)。2026-09-29 的 [50 题回答质量测试](docs/整理记录/版本0.2/第一轮回答质量/人工复核.md)已执行，自动结果为有答案题 24/42、正确拒答 6/8，整体验收未通过；逐题报告区分了真实回答缺漏、评分误判和题集标注问题。

题集与评分器已按首轮结果修订；首轮题集另存，当前仍为 50 题（43 题有答案、7 题应拒答）。[第二轮本地复核](docs/整理记录/版本0.2/第二轮检索/本地复核.md)记录了旧回答重评分和新的检索对照。随后经用户授权执行新版 50 题的向量与 BM25 Chat 回答质量测试，结果均未通过验收；详见[检索修改后测试结论](docs/整理记录/版本0.2/检索回归_2026-10-01/测试结论.md)。

2026-10-01 的五项验收边界修复及170项回归记录见 [整改记录 02](docs/整理记录/版本0.2/整改记录/02_项目复核与验收边界.md)。之后获授权完成真实Chat测试，2026-10-03 继续修复等义事实核验、预约手续漏答、无关来源混入及具体地点回答；采用模型选择原文短句编号、代码确定文本和逐句引用的协议。实施与最终结果见 [整改记录 03](docs/整理记录/版本0.2/整改记录/03_测试后回答质量与在线召回.md)。

2026-10-03最终验收：BM25与混合召回均为43/43有答案质量通过、7/7正确拒答，引用支持43/43、复核和错误均0；209项本地回归通过。本地配置已启用hybrid，V0.2本轮实现完成冻结。完整结果及范围见[验收结论](docs/整理记录/版本0.2/质量整改_2026-10-03/验收结论.md)。

离线 BM25 使用独立的 `--bm25-min-score` 门槛（默认原始分大于 0）；混合召回依据原始余弦或 BM25 达标判断，RRF 分只用于排序。报告保存各通道分数及配置。纯检索命中门槛通过且无错误时退出码为 0；`--quality` 还要求实际执行并通过全部质量验收，任何失败或待人工复核均返回 1。

## 当前接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | 返回服务、环境和版本信息 |
| POST | `/documents` | 上传 PDF、Markdown 或 TXT，解析并写入本地向量索引（最大 10 MiB） |
| POST | `/query` | 返回回答、拒答状态、引用来源和 Top-K 调试检索结果 |

## 当前限制

- PDF 仅提取已有文本层；扫描件 OCR、图片和复杂表格不在 V0.1 范围内。
- 拒答使用余弦相似度阈值，默认值为 `0.25`；应按所用 Embedding 模型和数据调整。
- 在线召回支持 `TRACERAG_RETRIEVAL_STRATEGY=vector|bm25|hybrid`；缺省为 vector。混合召回复用已有向量索引与 Chunk 文本，切换策略后重启 API，无需重新上传；同名更新后会刷新 BM25。`TRACERAG_BM25_MIN_SCORE` 默认 0，`TRACERAG_RRF_K` 默认 60。API 与界面区分余弦、BM25 和 RRF 排序分。暂不支持用户登录、知识库隔离、删除文档或 Reranker。
- 回答显示所选原文短句及逐句引用，保持来源语言；模型必须返回内部编号选择 JSON。协议错误会明确报错，不能回退到未经校验的自由文本。原文引用不能单独保证问题覆盖完整，仍需业务评测。
- FastAPI 没有认证授权，默认仅绑定 `127.0.0.1`；不要在未加访问控制时暴露到公网。
