# API 与 UI 使用同一镜像；模型权重由只读挂载提供。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/tracerag

# FAISS 的 Linux wheel 依赖 OpenMP 运行库。
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements-release.txt README.md ./
COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY samples ./samples

# 默认只安装在线 Embedding 所需依赖；本地模型构建时启用扩展。
ARG INSTALL_LOCAL_EMBEDDINGS=false
RUN python -m pip install --no-cache-dir -c requirements-release.txt "." \
    && if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
         python -m pip install --no-cache-dir -c requirements-release.txt ".[local-embeddings]"; \
       fi \
    && mkdir -p /var/lib/tracerag /models/embedding

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
