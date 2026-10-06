# TraceRAG

TraceRAG 是一个可溯源的知识库问答系统。当前包版本为 **0.2.1**，包含文档导入、向量/BM25/RRF 检索、原文回答、逐句引用、资料库管理与离线评测。

正式版本入口：[V0.2.1 发布](https://github.com/FTYFT-qqr/TraceRAG/releases/tag/v0.2.1)。该版本包含资料库管理、新版界面及独立业务评测工具；验收范围与限制见 [007 发布整改](docs/版本记录/V0.2/007_完成度检查与发布验收整改.md)。旧版发布与演示视频保留在 [V0.2.0](https://github.com/FTYFT-qqr/TraceRAG/releases/tag/v0.2.0)。

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

## 文档与交付入口

| 内容 | 入口 |
| --- | --- |
| 全部文档、目录与维护规则 | [文档中心](docs/README.md) |
| Docker API/UI 启动、本地模型挂载与索引持久化 | [容器部署](docs/使用指南/容器部署.md) |
| 在线链路与离线评测架构 | [系统架构](docs/使用指南/系统架构.md) |
| 固定演示、检索改进与简历表述 | [演示指南](docs/使用指南/演示指南.md) |
| 独立环境测试与剩余交付项 | [工程交付验收](docs/版本记录/V0.2/004_工程交付验收.md) |
| 版本范围与升级说明 | [V0.2 版本总结](docs/版本记录/V0.2/001_版本总结.md) |

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
  README.md           文档总入口与维护规则
  开发路线图.md       原定范围、状态及后续方向
  使用指南/           系统架构、容器部署及演示
  版本记录/           各版本总结、整改、评测与工程验收
  评测数据/           共用标准、证据索引与原始输出
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

Chat 服务需要 API 密钥：在 `.env` 中填写 `TRACERAG_API_KEY`，也可用已有的 `OPENAI_API_KEY` 环境变量。Embedding 默认使用 OpenAI 兼容 API；也可以切换为本地 Sentence Transformers 模型。没有 Chat API 密钥时 `/health` 仍可访问，但需要运行时的目录、上传、删除和问答接口会返回 503。

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

## 评测与验收

```powershell
python -m pytest
```

发布收尾已保存的 Windows/Linux 远程回归各 **222 项通过**，Docker 工程检查及三个远程 CI 任务全部通过；新增回归覆盖中文控制台编码和 Windows 子进程清理。容器真实模型与重启的 19 项检查属于此前独立批次。实际浏览器演示为 **5 分 33 秒**，覆盖公开资料上传、回答、引用、拒答和恢复；准确发布提交及其 CI 链接见[发布页面](https://github.com/FTYFT-qqr/TraceRAG/releases/tag/v0.2.0)，原始证据见[005 发布收尾](docs/版本记录/V0.2/005_发布收尾与下一阶段规划.md)。

固定 50 题最终质量验收：BM25/混合召回均为有答案质量 43/43、引用支持 43/43、正确拒答 7/7，复核与执行错误均为 0。这是三份公开语料的样本结果，不代表任意业务数据的准确率。[版本总结](docs/版本记录/V0.2/001_版本总结.md)、[评测标准](docs/评测数据/评测标准.md)与[原始证据](docs/评测数据/README.md)分别说明结论、口径和运行记录。

V0.1 真实模型最小验收为 Top-5 14/15、有答案质量 15/15、正确拒答 4/4，历史回归 91 项；详见[V0.1 总结](docs/版本记录/V0.1/001_版本总结.md)。

```powershell
# V0.1 最小真实模型验收
python scripts/evaluate_retrieval.py
# V0.2 五策略本地检索对照，使用新目录
python scripts/evaluate_v02.py --suite --output-dir docs/评测数据/版本0.2/临时运行_新轮次

# 006 新业务语料验收示例：每个策略使用独立的新输出目录
python scripts/evaluate_v02.py --corpus-dir samples/business/006 --questions samples/business/006/questions_acceptance_v2.json --dataset-mode acceptance --strategy hybrid --quality --name 业务泛化混合召回 --output-dir docs/评测数据/版本0.2/业务泛化_新轮次/混合
```

V0.2 评测默认使用 `samples/acceptance/` 的公开样例；`--corpus-dir` 可切换到新业务资料目录，配合 `--questions` 和 `--dataset-mode acceptance|development` 指定题集。评测在内存中建立隔离索引，不读取上传资料或写应用索引。V0.1 脚本固定写入 `docs/评测数据/版本0.1/验收记录/检索验收记录.md`，不提供输出路径参数；重跑前先备份历史文件，运行后将新结果另存至新批次目录并恢复历史原件。V0.2 默认只检索，增加 `--quality` 才执行真实 Chat 质量评测；问题及检索样例会发送给 `.env` 配置的 Chat 服务。`--suite --quality` 默认仅首轮做回答质量；如需所有轮次执行真实回答可加 `--quality-scope all`，或分别运行单策略。

006 的模拟业务资料 v2 完整 50 题已实测：BM25 与混合召回各通过 40/40 有答案题和 10/10 拒答题；纯向量有 1 题证据排在 Top-5 之外，质量为 39/40。三策略报告各包含真实 Chat、逐题引用和性能数据；详见[006 结果与边界](docs/版本记录/V0.2/006_业务泛化评测与性能基线.md)。这些模拟题的表现不能代表真实业务准确率。

BM25 使用独立原始分门槛，混合召回按原始通道判断是否允许生成，RRF 只排序。纯检索命中门槛满足且无错误时退出码为 0；质量模式还要求全部质量验收通过，失败或待复核返回 1。各轮失败、重评分及 Bad Case 见[评测与人工复核](docs/版本记录/V0.2/003_评测与人工复核.md)。

## 当前接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | 返回服务、环境和版本信息 |
| GET | `/documents` | 列出当前文档、片段数量、已索引页数和快照标识 |
| POST | `/documents` | 上传 PDF、Markdown 或 TXT，解析并写入本地向量索引（最大 10 MiB） |
| DELETE | `/documents/{document_id}` | 从当前知识库移除文档并提交新快照；历史快照仍保留 |
| POST | `/query` | 返回回答、拒答状态、引用来源和 Top-K 调试检索结果 |

当前工作区已实现资料库界面与列表、删除接口。新增列表/删除、前端交互及空快照测试后，当前源码完整回归为 **259 项通过**；隔离固定协议工程链路 **19/19**、新版候选镜像的容器链路 **10/10** 通过，准确输入与报告见[007 完成度与发布验收整改](docs/版本记录/V0.2/007_完成度检查与发布验收整改.md)。此前 V0.2.0 的 222 项回归保留为历史记录，不能替代当前源码验收。

## 当前限制

- PDF 仅提取已有文本层；当前尚未支持扫描件 OCR、图片和复杂表格。
- 向量策略按原始余弦分预筛选，默认门槛 `0.25`；BM25 和混合策略使用独立原始通道门槛，RRF 只排序。门槛应按模型和资料调整，不能单独证明答案正确。
- 在线召回支持 `TRACERAG_RETRIEVAL_STRATEGY=vector|bm25|hybrid`；缺省为 vector。混合召回复用已有向量索引与 Chunk 文本，切换策略后重启 API，无需重新上传；同名更新及删除后会刷新 BM25。`TRACERAG_BM25_MIN_SCORE` 默认 0，`TRACERAG_RRF_K` 默认 60。API 与界面区分余弦、BM25 和 RRF 排序分。暂不支持用户登录、知识库隔离或 Reranker。
- 回答显示所选原文短句及逐句引用，保持来源语言；模型必须返回内部编号选择 JSON。协议错误会明确报错，不能回退到未经校验的自由文本。原文引用不能单独保证问题覆盖完整，仍需业务评测。
- FastAPI 没有认证授权，默认仅绑定 `127.0.0.1`；不要在未加访问控制时暴露到公网。
