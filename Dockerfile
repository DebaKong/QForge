# QForge 应用镜像（可选部署方式；本机默认走原生安装，见 README）
#
# 两个构建目标（依赖边界不同，体积差很大）：
#   --target api    只装核心依赖（Web/队列/校验/预处理/代码生成），**不需要 GPU**，约 1 GB；
#   --target worker 额外装 GPU 依赖（TensorRT 运行库 + cuda-python），用于构建 Engine，
#                   Linux 版 TensorRT 运行库单包约 3.7 GB，因此镜像约 9 GB。
#   compose 里 api / worker 各用对应目标，镜像名不同（qforge-api / qforge-worker）。
#
# 其它设计要点：
#   1. 多阶段：Node 阶段构建前端，运行期不需要 Node；
#   2. 前端放在 /opt/qforge/web（**镜像内**）并用 QFORGE_FRONTEND_DIR 指定，
#      避免被 /data 数据卷遮蔽（卷挂载会覆盖镜像内同路径内容，是常见坑）；
#   3. 数据（任务目录、数据库）统一放 /data，作为卷持久化；
#   4. 镜像不装 CUDA/TensorRT **开发包**（头文件+导入库）：那一步是可选能力，
#      容器里默认缺失并在报告里如实标为 BLOCKED。
#
# syntax=docker/dockerfile:1

# 软件源（可按网络环境覆盖，例如国内镜像）：
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG NPM_REGISTRY=https://registry.npmjs.org

# ---------------- 前端构建 ----------------
FROM node:22-alpine AS frontend
ARG NPM_REGISTRY
WORKDIR /src/frontend
# 先装依赖再拷源码，利用层缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm config set registry "${NPM_REGISTRY}" \
 && npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------- 源码与元数据（供两个目标复用）----------------
FROM python:3.10-slim AS source
ARG PIP_INDEX_URL
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}
WORKDIR /opt/qforge
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
COPY workers/ ./workers/

# ---------------- 共同运行配置 ----------------
FROM source AS core
ENV QFORGE_FRONTEND_DIR=/opt/qforge/web \
    QFORGE_DATA_DIR=/data/storage \
    QFORGE_EXECUTOR_MODE=celery \
    QFORGE_CELERY_TASK_ALWAYS_EAGER=false \
    QFORGE_CELERY_BROKER_URL=redis://redis:6379/0 \
    QFORGE_CELERY_RESULT_BACKEND=redis://redis:6379/1
VOLUME ["/data"]
EXPOSE 8000

# ---------------- api：核心依赖（不含 GPU）----------------
FROM core AS api
RUN pip install . \
 && rm -rf /opt/qforge/backend /opt/qforge/workers /opt/qforge/pyproject.toml
COPY --from=frontend /src/frontend/dist /opt/qforge/web
CMD ["qforge", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-worker"]

# ---------------- worker：核心 + GPU 依赖 ----------------
FROM core AS worker
RUN pip install ".[gpu]" \
 && rm -rf /opt/qforge/backend /opt/qforge/workers /opt/qforge/pyproject.toml
COPY --from=frontend /src/frontend/dist /opt/qforge/web
CMD ["qforge", "worker", "--concurrency", "1"]
