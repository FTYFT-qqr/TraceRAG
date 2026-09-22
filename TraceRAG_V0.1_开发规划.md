# TraceRAG V0.1 开发规划

## 1. 版本定位

TraceRAG V0.1 定义为：**最小可运行 RAG 闭环**。

本版本的目标不是追求功能丰富，而是稳定跑通以下核心链路：

> 导入文档 → 切分文本 → 建立索引 → 用户提问 → 检索证据 → 基于证据回答 → 返回引用 → 证据不足时拒答

只要这条链路稳定、可验证、可追溯，V0.1 即视为成功。

---

## 2. 功能范围

| 功能 | V0.1 要实现 | 暂时不做 |
| --- | --- | --- |
| 文档导入 | PDF、Markdown、TXT | Word、网页、OCR |
| 文档解析 | 提取正文，PDF 保留页码 | 复杂表格解析 |
| 文本清洗 | 去除明显空白和异常换行 | 高级语义清洗 |
| Chunk | 固定长度切分 + overlap | Semantic Chunking |
| 元数据 | 文件名、页码、`chunk_id`、`chunk_index` | 标签体系 |
| Embedding | OpenAI 兼容接口 | 多模型动态选择 |
| 向量检索 | FAISS Top-K Search | BM25、Hybrid Search |
| RAG 回答 | LLM 仅依据检索内容回答 | Agent、联网搜索 |
| Citation | 返回来源文件、页码、Chunk | 精细句级引用 |
| 拒答 | 简单相关度阈值 + Prompt 约束 | 学习型置信度模型 |
| API | `/health`、`/documents`、`/query` | 用户管理等非核心接口 |
| UI | Streamlit 简单问答界面 | 正式 Web 前端 |
| 测试 | 核心模块基础 pytest | 完整测试覆盖率 |
| 部署 | 本地运行 | Docker、Kubernetes、云部署 |

---

## 3. 最终用户体验

### 3.1 有足够证据时

```text
打开 TraceRAG
    ↓
上传 employee_handbook.pdf
    ↓
系统解析文档并建立索引
    ↓
用户提问：公司的年假有多少天？
    ↓
系统回答：正式员工每年享有 10 天带薪年假。[1]

来源：
[1] employee_handbook.pdf · 第 12 页
```

### 3.2 证据不足时

```text
用户提问：公司的股票期权什么时候可以行权？

系统回答：根据当前知识库，我无法确认这个问题。
```

---

## 4. 开发阶段

V0.1 分为 8 个阶段，按顺序开发和验收，不同时铺开。

### 阶段 1：项目骨架

**目标：** 建立 `app/`、`tests/`、`data/`、配置文件和 FastAPI 入口。

**重点：** 先形成结构简单、能够运行的项目基础，不提前引入后续功能。

**验收标准：** 项目可以正常启动，`GET /health` 返回健康状态。

### 阶段 2：Document Loader

**目标：** 实现 PDF、Markdown、TXT 读取，并统一转换为内部数据结构。

**重点：** PDF 必须保留页码；所有文档必须保留文件名和来源信息。

**验收标准：** 可以输出每页或每段的文本、文件名和页码。

### 阶段 3：Chunk Pipeline

**目标：** 实现文本清洗、固定长度切块、overlap，并生成稳定的 `chunk_id`。

**重点：** 每个 Chunk 至少保存 `content`、`document_id`、`file_name`、`page_number` 和 `chunk_index`。

**验收标准：** 随机抽取任意 Chunk，均能追溯到原始文档及对应页码。

### 阶段 4：Embedding + FAISS

**目标：** 调用 OpenAI 兼容的 Embedding API，将 Chunks 向量化并建立 FAISS 索引。

**重点：** 索引与 Chunk 元数据的映射必须稳定，并支持持久化。

**验收标准：** 程序重启后可以加载已有索引，无需重复执行 Embedding。

### 阶段 5：Vector Retriever

**目标：** 实现统一的向量检索接口，例如 `retrieve(query, top_k=5)`。

**重点：** 返回 Chunk、元数据和相关度分数；本阶段不接入 LLM。

**验收标准：** 使用十几个测试问题验证时，正确证据大多数能够进入 Top-K。

### 阶段 6：RAG + Citation + Reject

**目标：** 将 Top-K Context 提供给 LLM，生成仅基于证据的回答，并实现引用和拒答。

**重点：** Citation 应对应实际使用的证据；当 Top-1 或 Top-K 相关度过低时拒答。

**验收标准：** 以下三种情况均通过：

- 有答案的问题能够正确回答；
- 回答附带可追溯来源；
- 没有足够证据的问题不会编造答案。

### 阶段 7：FastAPI + Streamlit

**目标：** 将已完成的 RAG Pipeline 封装为 API 和简单界面。

**重点：** API 仅保留 `GET /health`、`POST /documents`、`POST /query` 三个核心接口；Streamlit 仅负责文件上传、问题输入，以及回答、引用和调试检索结果的展示。

**验收标准：** 用户能够在浏览器中完成“上传文档 → 提问 → 查看回答与引用”的完整流程。

### 阶段 8：基础测试与 README

**目标：** 为核心流程补充基础自动化测试和项目说明。

**重点：** 测试至少覆盖 Chunk、Retrieval 和 API 基本流程；README 说明启动方法、项目架构、支持能力与当前限制。

**验收标准：** 核心测试可以稳定通过，新使用者能够按照 README 在本地启动项目。

---

## 5. 核心数据结构

V0.1 围绕以下四个结构设计，不提前建立大量 DTO。

### Chunk

```python
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    file_name: str
    page_number: int | None
    chunk_index: int
```

### RetrievalResult

```python
class RetrievalResult:
    chunk: Chunk
    score: float
```

### QueryResponse

```python
class QueryResponse:
    answer: str
    rejected: bool
    citations: list[Citation]
```

### Citation

```python
class Citation:
    chunk_id: str
    file_name: str
    page_number: int | None
    text: str
```

---

## 6. V0.1 明确不做

以下内容不属于 V0.1 范围：

- BM25、Hybrid Search、Reranker
- 完整 Evaluation Framework
- SQLite、SQLAlchemy
- 用户系统、登录注册、权限、知识库空间、多租户
- Celery、Redis、消息队列
- LangGraph、Agent、GraphRAG、知识图谱
- OCR、复杂表格、图片理解
- React、Vue
- Docker、Kubernetes、云部署

其中 BM25、Evaluation 和 Bad Case Analysis 计划放入 V0.2。推荐的后续迭代路径为：

```text
V0.1 Vector Baseline
    ↓
建立 Evaluation Dataset
    ↓
发现并分析 Bad Case
    ↓
加入 BM25 + RRF
    ↓
重新评测
    ↓
验证 Recall@K 是否提升
```

---

## 7. V0.1 验收与冻结标准

以下 8 项全部通过后，V0.1 即完成并冻结：

- [ ] PDF、Markdown、TXT 可以成功导入；
- [ ] PDF 页码能够正确保留；
- [ ] Chunk 可以追溯到原始文档；
- [ ] 文档可以完成 Embedding 并建立 FAISS 索引；
- [ ] Query 可以检索 Top-K Chunks；
- [ ] LLM 仅使用检索证据回答；
- [ ] 回答能够返回文件名和页码 Citation；
- [ ] 没有足够证据的问题能够拒答。

## 8. 开发原则

采用逐阶段开发和验收方式：**前一阶段验收通过后，再进入下一阶段。**

V0.1 达到上述 8 项冻结标准后，不再因为“功能看起来简单”而继续增加范围。下一版本再集中解决评测、Bad Case 分析和混合检索优化问题。
